from database_functions import *
from training_functions import *
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

CSL = CheckpointSaverLoader()
loaded = False
if os.listdir(CSL.main_dir):
    loaded = True
    load = CSL.load_checkpoint()
print("CSL loaded")

animals = ('cow','deer','sheep','zebra','elephant','tiger','rhino','giraffe')
train_prop = 0.7
valid_prop = 0.2

num_classes = len(animals)
ydims = 4
embedding_dims = 512
num_encoder_layers = 6
num_decoder_layers = 6
num_attn_heads = 8

weight_decay_bounds = [1e-8, 1e-3]
weight_decay_init = [1e-6, 1e-5]
HPO = HPOptimiser(weight_decay_init, weight_decay_bounds)
if loaded:
    HPO.load_state(load["hpo state"])

NUM_HPS = 10
NUM_EPOCHS = 150
batch_size = 64
early_stop_threshold = 10
hp_start = int(HPO.y_obs.size(0))

for hp_index in range(hp_start, NUM_HPS):

    HPO.calculate_gp_pars()
    wd = HPO.select_new_x()
    CSL.save_hp(wd, hp_index)

    dh = ODDataHandler(animals, train_prop, valid_prop)
    if loaded and load["dh state"] is not None:
        dh.load_state(load["dh state"])
    CSL.save_data_handler_state(dh.get_state())

    model = AnimalDetector(num_classes=num_classes, 
                           ydims=ydims, 
                           embedding_dims=embedding_dims,
                           num_encoder_layers=num_encoder_layers,
                           num_decoder_layers=num_decoder_layers,
                           num_attn_heads=num_attn_heads).to(device)
    
    if not loaded or load["loss scale params"] is None:
        log_sig_ce = torch.tensor(0., requires_grad=True, device=device)
        log_sig_mse = torch.tensor(0., requires_grad=True, device=device)
        log_sig_IoU = torch.tensor(0., requires_grad=True, device=device)
    else:
        log_sig_ce = torch.tensor(load["loss scale params"]["log_sig_ce"], requires_grad=True, device=device)
        log_sig_mse = torch.tensor(load["loss scale params"]["log_sig_mse"], requires_grad=True, device=device)
        log_sig_IoU = torch.tensor(load["loss scale params"]["log_sig_IoU"], requires_grad=True, device=device)

    optimiser = optim.Adam([{"params":model.parameters()}, 
                            {"params":log_sig_ce}, 
                            {'params':log_sig_mse}, 
                            {'params':log_sig_IoU}], 
                            lr = 1e-4, 
                            weight_decay=wd)
    
    if loaded and load["current model state"] is not None:
        model.load_state_dict(load["current model state"])
        optimiser.load_state_dict(load["current optimiser state"])

    print(f"Selected weight decay: {wd}")
    print(f"Optimiser weight decay: {optimiser.param_groups[0]["weight_decay"]}\n")
    early_stop_counter = 0 if not loaded or load["early stop threshold"] is None else load["early stop threshold"]
    best_vloss = torch.inf if not loaded or load["best vloss"] is None else load["best vloss"]
    epoch_start = 0 if not loaded or load["num epochs"] is None else load["num epochs"]

    print("Model loaded. \nStarting training...")
    if loaded:
        print(f"Loaded best vloss: {best_vloss:.3f}\nEarly stop counter: {early_stop_counter}")
    for epoch in range(epoch_start, NUM_EPOCHS):

        print(f"Epoch {epoch + 1}:\n")
        train_dl = DataLoader(dh.train_data, batch_size, shuffle=True)
        valid_dl = DataLoader(dh.valid_data, batch_size, shuffle=True)

        training_ce_losses = []
        training_mse_losses = []
        training_IoU_losses = []
        training_scale_parameters = []
        training_scaled_losses = []
        training_total_losses = []

        print("\tTraining loop:\n\n")
        for idx, batch in enumerate(train_dl):

            optimiser.zero_grad()
            ce_loss, mse_loss, IoU_loss = calculate_training_losses(device, model, dh, tuple(batch.tolist()), augment=True)
            scaled_loss = ce_loss / (2 * log_sig_ce.exp() ** 2) + log_sig_ce + \
                          mse_loss / (2 * log_sig_mse.exp() ** 2) + log_sig_mse + \
                          IoU_loss / (2 * log_sig_IoU.exp() ** 2) + log_sig_IoU
            total_loss = ce_loss + mse_loss + IoU_loss
            scaled_loss.backward()
            optimiser.step()

            training_ce_losses.append(ce_loss.item())
            training_mse_losses.append(mse_loss.item())
            training_IoU_losses.append(IoU_loss.item())
            training_scale_parameters.append([log_sig_ce.item(), log_sig_mse.item(), log_sig_IoU.item()])
            training_scaled_losses.append(scaled_loss.item())
            training_total_losses.append(total_loss.item())

            print_log = f"\t\tEpoch {epoch + 1} | Batch {idx + 1} | " \
                        f"CE: {training_ce_losses[-1]:.3f} | MSE: {training_mse_losses[-1]:.3f} | IoU: {training_IoU_losses[-1]:.3f} | " \
                        f"log CE scale: {log_sig_ce:.3f} | log MSE scale: {log_sig_mse:.3f} | log IoU scale: {log_sig_IoU:.3f} | " \
                        f"Scaled: {training_scaled_losses[-1]:.3f} | Total: {training_total_losses[-1]:.3f}"
            print(print_log)

        print("\n\tTraining completed!\n\n")
        total_ce_loss = 0
        total_tse_loss = 0
        total_IoU_loss = 0
        total_correct = 0
        total_tae = 0

        print(f"\tStarting validation, vloss to beat: {best_vloss:.3f}\n")
        for idx, batch in enumerate(valid_dl):

            ce_loss, tse_loss, IoU_loss, num_correct, tae = calculate_validation_losses(device, model, dh, tuple(batch.tolist()))

            total_ce_loss += ce_loss
            total_tse_loss += tse_loss
            total_IoU_loss += IoU_loss
            total_correct += num_correct
            total_tae += tae

            if (idx + 1) % 5 == 0 or (idx + 1) == len(valid_dl):
                print(f"\t\t{idx + 1}/{len(valid_dl)} completed")

        mce_loss_per_batch = total_ce_loss / len(valid_dl)
        mse_loss = total_tse_loss / dh.valid_size
        mIoU_loss_per_batch = total_IoU_loss / len(valid_dl)

        accuracy = total_correct / dh.valid_size
        mae = total_tae / dh.valid_size

        vloss = mce_loss_per_batch + mse_loss + mIoU_loss_per_batch
        print_log = f"\n\t\tEpoch {epoch + 1}\n" \
                    f"\t\tMCE per batch: {mce_loss_per_batch:.3f}\n\t\tMSE: {mse_loss:.3f}\n" \
                    f"\t\tMIoU per batch: {mIoU_loss_per_batch:.3f}\n\t\tAcc: {100 * accuracy:.2f}%\n\t\t" \
                    f"MAE: {mae:.3f}\n\t\ttotal vloss: {vloss:.3f}\n"
        print(print_log)

        CSL.save_training_losses(training_ce_losses, 
                                 training_mse_losses, 
                                 training_IoU_losses,
                                 training_scale_parameters, 
                                 training_scaled_losses, 
                                 training_total_losses)
        
        CSL.save_validation_losses(mce_loss_per_batch, mse_loss, mIoU_loss_per_batch, vloss)
        CSL.save_validation_metrics(accuracy, mae, mIoU_loss_per_batch)
        CSL.save_current_model_optimiser(model.state_dict(), optimiser.state_dict(), 
                                         [log_sig_ce.item(), log_sig_mse.item(), log_sig_IoU.item()])

        if vloss < best_vloss:
            print("\t\tNew best achieved!")
            CSL.save_best_model(model.state_dict())
            best_vloss = vloss
            early_stop_counter = 0
            CSL.save_early_stop_threshold(early_stop_counter)

        else:
            early_stop_counter += 1
            print(f"\t\tEarly stop count: {early_stop_counter}")
            CSL.save_early_stop_threshold(early_stop_counter)
            if early_stop_counter > early_stop_threshold:
                print("Stopped early")
                CSL.save_epoch(epoch + 1)
                CSL.save_early_stopping()
                break

        CSL.save_epoch(epoch + 1)
        print("\n\tCheckpoint saved!\n")

    print("Model finished training")
    HPO.observe_y(best_vloss)
    CSL.save_hpo_state(HPO.get_state())
    CSL.save_finished_training()