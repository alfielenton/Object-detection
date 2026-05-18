from database_functions import *
from networks.animal_detection import AnimalDetector
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

def run_training_step(model:AnimalDetector, optimiser:optim.Adam, data_handler:ODDataHandler, batch:tuple):

    with torch.enable_grad():

        optimiser.zero_grad()
        model.train()

        x_ims, xc_seqs, xb_seqs, yc, yb = data_handler.build_xy_batch(batch)

        ypcs, ypbs = model(x_ims, xc_seqs, xb_seqs)
        
        ce_loss = -ypcs[torch.arange(len(batch)),yc].sum(dim=0, keepdim=True)

        box_mask = yc != len(data_handler.animals)
        yb = yb[box_mask]
        ypbs = ypbs[box_mask]

        p_and_t = torch.stack([ypbs,yb])

        cs = p_and_t[..., :2]
        ds = p_and_t[..., 2:]

        mse_loss = ((cs[0, ...] - cs[1, ...])**2).sum(dim=1).mean(dim=0, keepdim=True)

        xmins = cs[..., 0] - .5 * ds[..., 0]
        xmaxs = cs[..., 0] + .5 * ds[..., 0]
        ymins = cs[..., 1] - .5 * ds[..., 1]
        ymaxs = cs[..., 1] + .5 * ds[..., 1]

        xinter_mins = xmins.max(dim=0).values
        xinter_maxs = xmaxs.min(dim=0).values
        yinter_mins = ymins.max(dim=0).values
        yinter_maxs = ymaxs.min(dim=0).values

        w_inter = nn.ReLU()(xinter_maxs - xinter_mins)
        h_inter = nn.ReLU()(yinter_maxs - yinter_mins)

        A_inter = w_inter * h_inter
        A_union = ds.prod(dim=2).sum(dim=0) - A_inter

        IoU_loss = (1 - A_inter / A_union).sum(dim=0, keepdim=True)

        total_loss = torch.cat((ce_loss, mse_loss, IoU_loss))
        normed_loss = ((total_loss - total_loss.mean()) / total_loss.std()).sum()

        normed_loss.backward()
        optimiser.step()

        return ce_loss.item(), mse_loss.item(), IoU_loss.item(), normed_loss.item()

def run_validation_batch(model:AnimalDetector, data_handler:ODDataHandler, batch:tuple):

    with torch.no_grad():

        model.eval()
        x_ims, xc_seqs, xb_seqs, yc, yb = data_handler.build_xy_batch(batch)

        ypc, ypb = model(x_ims, xc_seqs, xb_seqs)

        ce_loss = -ypc[torch.arange(len(batch)), yc].sum(dim=0, keepdim=True)

        box_mask = yc != len(data_handler.animals)
        yb = yb[box_mask]
        ypb = ypb[box_mask]

        p_and_t = torch.stack([ypb, yb])

        cs = p_and_t[..., :2]
        ds = p_and_t[..., 2:]

        tse_loss = ((cs[0, ...] - cs[1,...])**2).sum(dim=1).sum(dim=0, keepdim=True)

        xmins = cs[..., 0] - .5 * ds[..., 0]
        xmaxs = cs[..., 0] + .5 * ds[..., 0]
        ymins = cs[..., 1] - .5 * ds[..., 1]
        ymaxs = cs[..., 1] + .5 * ds[..., 1]

        xinter_mins = xmins.max(dim=0).values
        xinter_maxs = xmaxs.min(dim=0).values
        yinter_mins = ymins.max(dim=0).values
        yinter_maxs = ymaxs.min(dim=0).values

        w_inter = nn.ReLU()(xinter_maxs - xinter_mins)
        h_inter = nn.ReLU()(yinter_maxs - yinter_mins)

        A_inter = w_inter * h_inter
        A_union = ds.prod(dim=2).sum(dim=0) - A_inter

        IoU_loss = (1 - A_inter / A_union).sum(dim=0, keepdim=True)

        c_preds = ypc.max(dim=1).indices
        num_correct = (c_preds == yc).sum()

        tae = torch.sqrt(((cs[0,...] - cs[1,...])**2).sum(dim=1)).sum(dim=0, keepdim=True)

        return ce_loss.item(), tse_loss.item(), IoU_loss.item(), num_correct.item(), tae.item()

class HPOptimiser:

    def __init__(self, x_inits, log_length_init, log_sigf_init, log_sign_init, pars_threshold):

        self.log_length = torch.tensor(log_length_init, dtype=torch.float32)
        self.log_sigf = torch.tensor(log_sigf_init, dtype=torch.float32)
        self.log_sign = torch.tensor(log_sign_init, dtype=torch.float32)

        self.pars_threshold = pars_threshold
        
        self.x_obs = torch.tensor(x_inits)

        self.y_obs = torch.tensor([])

    def observe_y(self, y):
        if self.x_obs.size(0) > self.y_obs.size(0):
            self.y_obs = torch.cat((self.y_obs, torch.tensor([y]))).to(torch.float32)
        else:
            raise ValueError("Number of y observations must be less than x observations to dd new observations")

    def calculate_gp_pars(self):
        
        with torch.enable_grad():
            curr_pars = [self.log_length.requires_grad_(True), 
                         self.log_sigf.requires_grad_(True),
                         self.log_sign.requires_grad_(True)]
            
            par_optim = optim.Adam(curr_pars)
            par_diff = torch.inf

            while par_diff > self.pars_threshold:

                par_optim.zero_grad()

                old_pars = [p.detach().clone() for p in curr_pars]

                l = torch.exp(self.log_length)
                sigf = torch.exp(self.log_sigf)
                sign = torch.exp(self.log_sign)

                K_y = sigf * torch.exp(-((self.x_obs - self.x_obs[..., None]) ** 2) / (2 * l ** 2)) \
                    + sign * torch.eye(self.x_obs.size(0))
                
                L_ky = torch.linalg.cholesky(K_y)

                K_y_inv_y =  torch.linalg.solve(L_ky, torch.linalg.solve(L_ky.T, self.y_obs))

                log_marg_likelihood = -.5 * self.y_obs @ K_y_inv_y - torch.log(torch.diag(L_ky)).sum()
                neg_log_marg_likelihood = -log_marg_likelihood

                neg_log_marg_likelihood.backward()
                par_optim.step()

                par_diff = ((torch.tensor(old_pars) - torch.tensor(curr_pars))**2).sum()
                print(par_diff)

        self.log_length, self.log_sigf, self.log_sign = curr_pars

    def select_new_x(self):
        pass