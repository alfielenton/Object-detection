from database_functions import *
from networks.animal_detection import AnimalDetector
import json
import os
import torch
from torch import nn
from torch import optim
from torch.distributions import Uniform

def calculate_training_losses(device, model:AnimalDetector, data_handler:ODDataHandler, batch:tuple, augment=True):

    with torch.enable_grad():

        model.train()

        x_ims, xc_seqs, xb_seqs, yc, yb = data_handler.build_xy_batch(batch, augment=augment)

        x_ims = x_ims.to(device)
        yc = yc.to(device)
        yb = yb.to(device)

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

        return ce_loss, mse_loss, IoU_loss

def calculate_validation_losses(device, model:AnimalDetector, data_handler:ODDataHandler, batch:tuple):

    with torch.no_grad():

        model.eval()
        x_ims, xc_seqs, xb_seqs, yc, yb = data_handler.build_xy_batch(batch, augment=False)

        x_ims = x_ims.to(device)
        yc = yc.to(device)
        yb = yb.to(device)

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

    def __init__(self, x_inits, x_bounds, kappa = 2.5, pars_threshold = 1e-6):

        self.log_length = torch.tensor(0., dtype=torch.float32)
        self.log_sigf = torch.tensor(0., dtype=torch.float32)
        self.log_sign = torch.tensor(0., dtype=torch.float32)

        self.kappa = kappa
        self.x_threshold = .1 * (x_bounds[1] - x_bounds[0])
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
        
        y_normed = (self.y_obs - self.y_obs.mean()) / self.y_obs.std()
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

                alpha = torch.linalg.solve(L_ky, y_normed)
                K_y_inv_y =  torch.linalg.solve(L_ky.T, alpha)

                log_marg_likelihood = -.5 * y_normed @ K_y_inv_y - torch.log(torch.diag(L_ky)).sum()
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
        self.x_threshold = .1 * (self.x_bounds[1] - self.x_bounds[0])

    def select_new_x(self, verbose : bool = False):

        lr = .25 * torch.abs(torch.tensor(self.x_bounds[0]) - torch.tensor(self.x_bounds[1]))
        y_normed = (self.y_obs - self.y_obs.mean()) / self.y_obs.std()

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

                alpha = torch.linalg.solve(LK_y, y_normed)
                K_y_inv_y = torch.linalg.solve(LK_y.T, alpha)

                mu_star = K_x_star_x @ K_y_inv_y

                alpha = torch.linalg.solve(LK_y, K_x_star_x)
                K_y_inv_K_x_star = torch.linalg.solve(LK_y.T, alpha)

                Sig_star = torch.sqrt(K_x_star_x_star - K_x_star_x @ K_y_inv_K_x_star)

                LCB = mu_star - self.kappa * Sig_star
                LCB.backward()
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
        self.curr_hp_index = max([int(dir[3:]) for dir in os.listdir(self.main_dir) if dir != "hpo_state.json"])
        self.curr_hp = "hp_" + str(self.curr_hp_index)
    
    def save_data_handler_state(self, state):

        datasets_path = self.main_dir + "//" + self.curr_hp + "//datasets//data.json"
        with open(datasets_path, "w") as f:
            json.dump(state, f)

    def load_data_handler_state(self):

        datasets_path = self.main_dir + "//" + self.curr_hp + "//datasets//data.json"
        if os.path.exists(datasets_path):
            with open(datasets_path, "r") as f:
                state = json.load(f)
        else:
            return None

        return state

    def save_hpo_state(self, state):

        with open(self.main_dir + "//hpo_state.json", "w") as f:
            json.dump(state, f)

    def load_hpo_state(self):

        with open(self.main_dir + "//hpo_state.json", "r") as f:
            state = json.load(f)
        return state

    def save_hp(self, hp:float, hp_index:int):

        self.get_curr_hp()
        hp_folder = self.main_dir + f"//hp_{hp_index}"

        if not os.path.exists(hp_folder):
            os.mkdir(hp_folder)

        with open(hp_folder + "//hp.txt", "w") as f:
            f.write(str(hp))

        with open(hp_folder + "//hp.txt", "w") as f:
            f.write(str(hp))

        if not os.path.exists(hp_folder + "//models"):
            os.mkdir(hp_folder + "//models")
        
        if not os.path.exists(hp_folder + "//metrics"):
            os.mkdir(hp_folder + "//metrics")

        if not os.path.exists(hp_folder + "//datasets"):
            os.mkdir(hp_folder + "//datasets")

        if not os.path.exists(hp_folder + "//metrics//metrics.json"):
            with open(hp_folder + "//metrics//metrics.json", "w") as f:
                metrics = {"finished training": False, 
                           "stopped early":False}
                
                json.dump(metrics, f)

    def save_current_model_optimiser(self, model_state_dict, optim_state_dict, loss_scale_params):

        model_dir = self.main_dir + "//" + self.curr_hp + "//models"
        torch.save(model_state_dict, model_dir + "//current_model.pth")
        torch.save(optim_state_dict, model_dir + "//optim.pth")

        with open(model_dir + "//loss_scale_params.json", "w") as f:
            data = {"log_sig_ce":loss_scale_params[0], 
                    "log_sig_mse":loss_scale_params[1], 
                    "log_sig_IoU":loss_scale_params[2]}
            json.dump(data, f)

    def load_current_model_optimiser(self):

        model_dir = self.main_dir + "//" + self.curr_hp + "//models"
        if os.listdir(model_dir):
            current_model_state = torch.load(model_dir + "//current_model.pth")
            current_optimiser_state = torch.load(model_dir + "//optim.pth")

            with open(model_dir + "//loss_scale_params.json", "r") as f:
                loss_scale_params = json.load(f)

            return current_model_state, current_optimiser_state, loss_scale_params
        else:
            return None, None, None
     
    def save_best_model(self, model_state_dict):

        model_dir = self.main_dir + "//" + self.curr_hp + "//models"
        torch.save(model_state_dict, model_dir + "//best_model.pth")

    def save_training_losses(self, ce_loss, mse_loss, IoU_loss, scaled_loss_params, scaled_loss, total_loss):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        if "training losses" in metrics.keys():
            metrics["training losses"]["ce loss"].append(ce_loss)
            metrics["training losses"]["mse loss"].append(mse_loss)
            metrics["training losses"]["IoU loss"].append(IoU_loss)
            metrics["training losses"]["scaled loss params"].append(scaled_loss_params)
            metrics["training losses"]["scaled loss"].append(scaled_loss)
            metrics["training losses"]["total loss"].append(total_loss)
        else:
            metrics["training losses"] = dict()
            metrics["training losses"]["ce loss"] = [ce_loss]
            metrics["training losses"]["mse loss"] = [mse_loss]
            metrics["training losses"]["IoU loss"] = [IoU_loss]
            metrics["training losses"]["scaled loss params"] = [scaled_loss_params]
            metrics["training losses"]["scaled loss"] = [scaled_loss]
            metrics["training losses"]["total loss"] = [total_loss]


        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def save_epoch(self, epoch):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["num epochs"] = epoch

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def load_epoch(self):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        if "num epochs" in metrics.keys():
            return metrics["num epochs"]
        else:
            return 0
    
    def save_early_stopping(self):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["stopped early"] = True
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def save_early_stop_threshold(self, threshold):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics['early stop threshold'] = threshold

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def load_early_threshold(self):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        if "early stop threshold" in metrics.keys():
            threshold = metrics['early stop threshold']
            return threshold
        else:
            return 0

    def save_finished_training(self):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        metrics["finished training"] = True
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def save_validation_losses(self, ce_loss, mse_loss, IoU_loss, total_loss):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        if "validation losses" in metrics.keys():
            metrics["validation losses"]["ce loss"].append(ce_loss)
            metrics["validation losses"]["mse loss"].append(mse_loss)
            metrics["validation losses"]["IoU loss"].append(IoU_loss)
            metrics["validation losses"]["total loss"].append(total_loss)
        else:
            metrics["validation losses"] = dict()
            metrics["validation losses"]["ce loss"] = [ce_loss]
            metrics["validation losses"]["mse loss"] = [mse_loss]
            metrics["validation losses"]["IoU loss"] = [IoU_loss]
            metrics["validation losses"]["total loss"] = [total_loss]

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def save_validation_metrics(self, accuracy, mae, IoU):

        metrics_path = self.main_dir + "//" + self.curr_hp + "//metrics//metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        if "validation metrics" in metrics.keys():
            metrics["validation metrics"]["accuracy"].append(accuracy)
            metrics["validation metrics"]["mean absolute error"].append(mae)
            metrics["validation metrics"]["mean IoU loss"].append(IoU)
        else:
            metrics["validation metrics"] = dict()
            metrics["validation metrics"]["accuracy"] = [accuracy]
            metrics["validation metrics"]["mean absolute error"] = [mae]
            metrics["validation metrics"]["mean IoU loss"] = [IoU]

        with open(metrics_path, "w") as f:
            json.dump(metrics, f)

    def load_checkpoint(self):

        hp_folder = self.main_dir + "//" + self.curr_hp
        with open(hp_folder + "//metrics//metrics.json", "r") as f:
            metrics = json.load(f)
        
        finished_training = metrics["finished training"]

        if os.path.exists(hp_folder) and not finished_training:
            metrics_path = hp_folder + "//metrics//metrics.json"
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            num_epochs = self.load_epoch()
            best_vloss = min(metrics["validation losses"]["total loss"]) if "validation losses" in metrics.keys() else torch.inf
            hpo_state = self.load_hpo_state()


            if os.path.exists(self.main_dir + "//" + self.curr_hp + "//models"):
                current_model_state_dict, current_optimiser_state_dict, loss_scale_params = self.load_current_model_optimiser()
            else:
                current_model_state_dict, current_optimiser_state_dict, loss_scale_params = None, None, None
            dh_state = self.load_data_handler_state()

            early_stop_threshold = self.load_early_threshold()

            return {"current model state" : current_model_state_dict,
                    "current optimiser state" : current_optimiser_state_dict,
                    "early stop threshold" : early_stop_threshold, 
                    "num epochs" : num_epochs, 
                    "best vloss" : best_vloss, 
                    "hpo state" : hpo_state, 
                    "loss scale params": loss_scale_params, 
                    "dh state" : dh_state}
        else:
            hpo_state = self.load_hpo_state()
            return {"current model state": None, 
                    "current optimiser state": None, 
                    "early stop threshold": None, 
                    "num epochs": None, 
                    "best vloss": None, 
                    "hpo state": hpo_state, 
                    "loss scale params": None, 
                    "dh state": None}