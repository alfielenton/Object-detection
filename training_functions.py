from database_functions import *
from networks.animal_detection import AnimalDetector
import json
import os
import torch
from torch import nn
from torch import optim
from torch.distributions import Uniform

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

        total_loss = ce_loss + mse_loss + IoU_loss

        total_loss.backward()
        optimiser.step()

        return ce_loss.item(), mse_loss.item(), IoU_loss.item(), total_loss.item()

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

    def __init__(self, x_inits, x_bounds, kappa = .1, pars_threshold = 1e-6, x_threshold = 1e-6):

        self.log_length = torch.tensor(0., dtype=torch.float32)
        self.log_sigf = torch.tensor(0., dtype=torch.float32)
        self.log_sign = torch.tensor(0., dtype=torch.float32)

        self.kappa = kappa
        self.x_threshold = x_threshold
        self.pars_threshold = pars_threshold
        self.x_bounds = x_bounds
        
        self.x_obs = torch.cat((torch.tensor(x_bounds), torch.tensor([.5 * (x_bounds[0] + x_bounds[1])])))
        if x_inits is not None:
            self.x_obs = torch.cat((torch.tensor(x_inits), self.x_obs))
        
        self.y_obs = torch.tensor([])

    def observe_y(self, y):
        if self.x_obs.size(0) > self.y_obs.size(0):
            self.y_obs = torch.cat((self.y_obs, torch.tensor([y]))).to(torch.float32)
        else:
            raise ValueError("Number of y observations must be less than x observations to add new observations")

    def calculate_gp_pars(self, verbose:bool = False):

        if self.x_obs.size(0) != self.y_obs.size(0):
            return
        
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

                K_y = sigf * torch.exp(-((self.x_obs - self.x_obs[..., None]) ** 2) / (2 * l)) \
                    + sign * torch.eye(self.x_obs.size(0))
                
                L_ky = torch.linalg.cholesky(K_y)

                alpha = torch.linalg.solve(L_ky, self.y_obs)
                K_y_inv_y =  torch.linalg.solve(L_ky.T, alpha)

                log_marg_likelihood = -.5 * self.y_obs @ K_y_inv_y - torch.log(torch.diag(L_ky)).sum()
                neg_log_marg_likelihood = -log_marg_likelihood

                neg_log_marg_likelihood.backward()
                par_optim.step()

                with torch.no_grad():
                    par_diff = torch.sqrt(((torch.tensor(old_pars) - torch.tensor(curr_pars))**2).sum())

                if verbose:
                    print(f"Log length: {self.log_length}")
                    print(f"Log sigf: {self.log_sigf}")
                    print(f"Log sign: {self.log_sign}")
                    print(f"Parameter change: {par_diff}\n\n")

        self.log_length.detach()
        self.log_sigf.detach()
        self.log_sign.detach()

    def get_state(self):

        state = {"x bounds": self.x_bounds, 
                 "x obs":self.x_obs.tolist(), 
                 "y obs": self.y_obs.tolist(), 
                 "kappa": self.kappa, 
                 "log length":float(self.log_length), 
                 "log sigf":float(self.log_sigf), 
                 "log sign":float(self.log_sign),
                 "pars threshold": float(self.pars_threshold),
                 "x threshold":float(self.x_threshold)}
        
        return state
    
    def load_state(self, state):

        self.x_bounds = state["x bounds"]
        self.x_obs = torch.tensor(state["x obs"])
        self.y_obs = torch.tensor(state["y obs"])
        self.kappa = state["kappa"]
        self.log_length = state["log length"]
        self.log_sigf = state["log sigf"]
        self.log_sign = state["log sign"]
        self.pars_threshold = state["pars threshold"]
        self.x_threshold = state["x threshold"]

    def select_new_x(self, verbose : bool = False):

        lr = .25 * torch.abs(torch.tensor(self.x_bounds[0]) - torch.tensor(self.x_bounds[1]))

        if self.x_obs.size(0) != self.y_obs.size(0):
            return self.x_obs[self.y_obs.size(0)]
        
        uniform = Uniform(self.x_bounds[0], self.x_bounds[1])
        with torch.enable_grad():

            x = uniform.sample().requires_grad_(True)
            x_optim = optim.Adam([x], lr)
            x_diff = torch.inf

            while x_diff > self.x_threshold:

                x_optim.zero_grad()
                x_old = x.detach().clone()

                l = torch.exp(self.log_length)
                sigf = torch.exp(self.log_sigf)
                sign = torch.exp(self.log_sign)

                K_y = sigf * torch.exp(-((self.x_obs - self.x_obs[..., None]) ** 2) / (2 * l)) \
                      + sign * torch.eye(self.x_obs.size(0))
                LK_y = torch.linalg.cholesky(K_y)
                
                K_x_star_x = sigf * torch.exp(-((self.x_obs - x) ** 2) / (2 * l))
                K_x_star_x_star = sigf

                alpha = torch.linalg.solve(LK_y, self.y_obs)
                K_y_inv_y = torch.linalg.solve(LK_y.T, alpha)

                mu_star = K_x_star_x @ K_y_inv_y

                alpha = torch.linalg.solve(LK_y, K_x_star_x)
                K_y_inv_K_x_star = torch.linalg.solve(LK_y.T, alpha)

                Sig_star = torch.sqrt(K_x_star_x_star - K_x_star_x @ K_y_inv_K_x_star)

                LCB = mu_star - self.kappa * Sig_star
                (-LCB).backward()
                x_optim.step()

                if x < self.x_bounds[0] or x > self.x_bounds[1]:
                    x = x_old
                    if verbose:
                        print(f"X chosen outside of bounds. Breaking")
                    break

                x_diff = torch.abs(x_old - x)
                if verbose:
                    print(f"X: {x}")
                    print(f"X difference: {x_diff}\n\n")

        self.x_obs = torch.cat((self.x_obs.detach(), torch.tensor([x])))
        return x.detach()
    

class CheckpointSaverLoader:

    def __init__(self):
        
        self.main_dir = "checkpoints"
        if os.listdir(self.main_dir):
            self.get_curr_hp()

    def get_curr_hp(self):
        hps = [dir for dir in os.listdir(self.main_dir) if dir != "hpo_state.json"]
        curr_hp = hps[0]
        index = int(curr_hp.replace("hp_", ""))

        for hp in hps:
            n_index = int(hp.replace("hp_", ""))
            if index < n_index:
                curr_hp = hp
                index = n_index

        self.curr_hp = curr_hp
    
    def save_data_handler_state(self, state):

        datasets_path = self.main_dir + "//" + self.curr_hp + "//datasets//data.json"
        with open(datasets_path, "w") as f:
            json.dump(state, f)

    def save_hpo_state(self, state):

        with open(self.main_dir + "//hpo_state.json", "w") as f:
            json.dump(state, f)

    def save_hp(self, hp:float, hp_index:int):

        self.curr_hp = f"hp_{hp_index}"
        hp_folder = self.main_dir + f"//hp_{hp_index}"
        os.mkdir(hp_folder)

        with open(hp_folder + "//hp.txt", "w") as f:
            f.write(str(hp))

        os.mkdir(hp_folder + "//models")
        os.mkdir(hp_folder + "//metrics")
        os.mkdir(hp_folder + "//datasets")

        with open(hp_folder + "//metrics//metrics.json", "w") as f:
            metrics = {"finished training": False, 
                       "stopped early": None, 
                       "num epochs": None, 
                       "training losses": {"ce loss": [], 
                                           "mse loss": [], 
                                           "IoU loss": [],
                                           "scaled loss":[], 
                                           "total loss": []},
                       "validation losses": {"ce loss": [], 
                                           "mse loss": [], 
                                           "IoU loss": [],
                                           "scaled loss":[], 
                                           "total loss": []}, 
                       "validation metrics": {"accuracy": [], 
                                              "mean absolute error":[],
                                              "mean IoU loss":[]}}
            
            json.dump(metrics, f)

    def save_current_model_optimiser(self, model_state_dict, optim_state_dict):

        model_dir = self.main_dir + "//" + self.curr_hp + "//models"
        torch.save(model_state_dict, model_dir + "//current_model.pth")
        torch.save(optim_state_dict, model_dir + "//optim.pth")

    def save_best_model(self, model_state_dict):

        model_dir = self.main_dir + "//" + self.curr_hp + "//models"
        torch.save(model_state_dict, model_dir + "//best_model.pth")

    def save_training_losses(self, ce_loss, mse_loss, IoU_loss, scaled_loss, total_loss):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["training losses"]["ce loss"].append(ce_loss)
        metrics["training losses"]["mse loss"].append(mse_loss)
        metrics["training losses"]["IoU loss"].append(IoU_loss)
        metrics["training losses"]["scaled loss"].append(scaled_loss)
        metrics["training losses"]["total loss"].append(total_loss)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def save_validation_losses(self, ce_loss, mse_loss, IoU_loss, scaled_loss, total_loss):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["validation losses"]["ce loss"].append(ce_loss)
        metrics["validation losses"]["mse loss"].append(mse_loss)
        metrics["validation losses"]["IoU loss"].append(IoU_loss)
        metrics["validation losses"]["scaled loss"].append(scaled_loss)
        metrics["validation losses"]["total loss"].append(total_loss)

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def save_validation_metrics(self, accuracy, mae, IoU):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["validation metrics"]["accuracy"].append(accuracy)
        metrics["validation metrics"]["mean absolute error"].append(mae)
        metrics["validation metrics"]["mean IoU loss"].append(IoU)

    def load_checkpoint(self):

        pass