from database_functions import *
from networks.counting_animals import AnimalCounter
import torch
from torch import optim
from torch.utils.data import DataLoader
from performance_tracker import PerformanceTracker

device = 'cuda'
N_MODELS = 1
dataset_size = 5000
train_prop = 0.8
valid_prop = None
valid = valid_prop is not None
training_batch_size = 32
valid_batch_size = 32
test_batch_size = 32
lr = 1e-5
weight_decay = 0.
N_EPOCHS = 5

pt = PerformanceTracker(valid)

for model in range(N_MODELS):

    if valid:
        train, valid, test = make_model_data(train_prop, valid_prop, dataset_size)
    else:
        train, test = make_model_data(train_prop, valid_prop, dataset_size)

    animal_abacus = AnimalCounter().to(device)
    aa_optimiser = optim.Adam(animal_abacus.parameters(), lr=lr, weight_decay=weight_decay)
    train_dl = DataLoader(train, training_batch_size)


    for epoch in range(N_EPOCHS):

        pt.start_epoch()
        
        for x_batch, y_batch in train_dl:

            if valid:
                valid_batch = random.sample(valid, valid_batch_size)
                x_valid_batch = torch.tensor(build_image_batch([x for x, _ in valid_batch]), dtype=torch.float32, device=device)
                y_valid_batch = torch.tensor([y for _, y in valid_batch], dtype=torch.float32, device=device)

            pt.start_time_component()
            test_batch = random.sample(test, test_batch_size)
            x_test_batch = torch.tensor(build_image_batch([x for x, _ in test_batch]), dtype=torch.float32, device=device)
            y_test_batch = torch.tensor([y for _, y in test_batch], dtype=torch.float32, device=device)
            time_taken = pt.end_time_component()
            print("\nBuilding test batch time: ", time_taken, "s")

            aa_optimiser.zero_grad()
            
            pt.start_time_component()
            x_batch = torch.tensor(build_image_batch(x_batch), dtype=torch.float32, device=device)
            y_batch = y_batch.to(dtype=torch.float32, device=device)
            time_taken = pt.end_time_component()
            print("Building train batch: ", time_taken, "s")

            pt.start_time_component()
            sig_hat = animal_abacus(x_batch)
            lambda_hat = (1 / sig_hat).squeeze()
            L = (lambda_hat - y_batch * torch.log(lambda_hat)).sum()
            pt.record_loss(L.detach(), 'train')

            L.backward()
            aa_optimiser.step()
            time_taken = pt.end_time_component()
            print("Gradient descent step: ", time_taken, "s")

            if valid:
                sig_valid_hat = animal_abacus(x_valid_batch)
                lambda_valid_hat = (1 / sig_valid_hat).squeeze()
                L_valid = (lambda_valid_hat - y_valid_batch * torch.log(lambda_valid_hat)).sum()
                pt.record_loss(L_valid.detach(), 'valid')

            pt.start_time_component()
            sig_test_hat = animal_abacus(x_test_batch)
            lambda_test_hat = (1 / sig_test_hat).squeeze()
            ae = torch.abs(lambda_test_hat - y_test_batch)
            mae = ae.mean()
            pt.record_test_performance(mae)
            time_taken = pt.end_time_component()
            print("Performing testing time: ", time_taken, "s\n")

            pt.summary_stats()

        pt.end_epoch()

    pt.save_model(animal_abacus, f"models//model-{model + 1}.pth")
    pt.save_model_results(f"models//model-{model + 1}-results.json")