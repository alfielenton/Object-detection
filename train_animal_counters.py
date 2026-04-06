from database_functions import *
from networks.counting_animals import AnimalCounter
import torch
from torch import optim
from torch.utils.data import DataLoader


N_MODELS = 5
dataset_size = 5000
train_prop = 0.8
valid_prop = None
valid = valid_prop is not None
batch_size = 32
lr = 1e-5
weight_decay = 0.
N_EPOCHS = 50

for model in range(N_MODELS):

    if valid_prop is None:
        train, test = make_model_data(train_prop, valid_prop, dataset_size)
    else:
        train, valid, test = make_model_data(train_prop, valid_prop, dataset_size)

    animal_abacus = AnimalCounter()
    aa_optimiser = optim.Adam(animal_abacus.parameters(), lr=lr, weight_decay=weight_decay)
    train_dl = DataLoader(train)


    for epoch in range(N_EPOCHS):

        
        for x_batch, y_batch in train_dl:

            aa_optimiser.zero_grad()
            
            x_batch = torch.tensor(build_image_batch(x_batch), dtype=torch.float32)
            y_batch = torch.tensor(y_batch, dtype=torch.float32)

            sig_hat = animal_abacus(x_batch)
            lambda_hat = 1 / sig_hat
            L = lambda_hat - y_batch * torch.log(lambda_hat)

            L.backward()
            aa_optimiser.step()


