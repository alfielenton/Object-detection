from database_functions import *
from networks.counting_animals import SingularityCheck
import torch
from torch import optim
from torchvision import transforms as T
from torch.utils.data import DataLoader
from performance_tracker import PerformanceTracker

device = 'cuda'
N_MODELS = 3
dataset_size = 5000
train_prop = 0.77
valid_prop = 0.12
valid = valid_prop is not None
training_batch_size = 32
valid_batch_size = 32
test_batch_size = 32
lr = 5e-4
weight_decay = 7.5e-7
early_stopping_threshold = 6
N_EPOCHS = 150

pt = PerformanceTracker(valid)
training_transform = T.Compose([T.RandomHorizontalFlip(),
                                T.RandomRotation(15),
                                T.RandomVerticalFlip()])

for model in range(N_MODELS):

    print(f'Model {model + 1} / {N_MODELS}\n\n')
    print('Data splits:\n')

    if valid:
        train, valid, test = make_model_data(train_prop, valid_prop)
        split_types = ['Training', 'Validation', 'Testing']
        splits = [np.array(train), np.array(valid), np.array(test)]

        for st, s in zip(split_types, splits):
            print(f'\t--{st}--')
            print(f'\tSize: {s.shape[0]}')
            print(f'\t0 and 1 split resp: {(s[:,1] == 0).sum() / s.shape[0]:.3f} -- {(s[:,1] == 1).sum() / s.shape[0]:.3f}\n')
    else:
        train, test = make_model_data(train_prop, valid_prop)
        split_types = ['Training', 'Testing']
        splits = [np.array(train), np.array(test)]

        for st, s in zip(split_types, splits):
            print(f'\t--{st}--')
            print(f'\tSize: {s.shape[0]}')
            print(f'\t0 and 1 split resp: {(s[:,1] == 0).sum() / s.shape[0]:.3f} -- {(s[:,1] == 1).sum() / s.shape[0]:.3f}\n')

    single_check = SingularityCheck().to(device)
    aa_optimiser = optim.Adam(single_check.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(N_EPOCHS):

        pt.start_epoch()
        
        print('\nTraining loop: \n')

        train_dl = DataLoader(train, training_batch_size, shuffle=True)
        single_check.train()
        for x_batch, y_batch in train_dl:

            aa_optimiser.zero_grad()
            
            x_batch = build_image_batch(x_batch).to(dtype=torch.float32, device=device)
            x_batch = torch.stack([training_transform(img) for img in x_batch])

            y_batch = y_batch.to(dtype=torch.float32, device=device)

            logits = single_check(x_batch).squeeze()
            L = torch.nn.BCEWithLogitsLoss(reduction='sum')(logits, y_batch)
            pt.record_loss(L.item(), 'train')

            L.backward()
            aa_optimiser.step()

            pt.end_forward_pass()
            pt.summary_stats('training')

        single_check.eval()
        if valid:
            print('\nValidation loop: \n')
            valid_dl = DataLoader(valid, valid_batch_size, shuffle=False)
            with torch.no_grad():
                for x_batch, y_batch in valid_dl:

                    x_batch = build_image_batch(x_batch).to(dtype=torch.float32, device=device)
                    y_batch = y_batch.to(dtype=torch.float32, device=device)

                    logits = single_check(x_batch).squeeze()
                    
                    L = torch.nn.BCEWithLogitsLoss(reduction='sum')(logits, y_batch)

                    pt.record_loss(L.item(), 'valid')
                    pt.end_forward_pass()
                    pt.summary_stats('validation')

        print('\nPerforming test')

        total_count = 0
        correct_count = 0
        test_dl = DataLoader(test, test_batch_size, shuffle=False)
        with torch.no_grad():
            for x_batch, y_batch in test_dl:

                x_batch = build_image_batch(x_batch).to(dtype=torch.float32, device=device)
                y_batch = y_batch.to(dtype=torch.float32, device=device)

                sigs = torch.nn.Sigmoid()(single_check(x_batch).squeeze())
                corr = ((sigs > 0.5) == y_batch).sum()
                tot = x_batch.size(0)

                correct_count += corr
                total_count += tot

            pt.record_test_performance((correct_count / total_count).item())
        
        pt.end_epoch()
        early_stop = pt.check_early_stopping(early_stopping_threshold)
        if early_stop:
            print(f"Stopping early at epoch {epoch + 1}")
            break

    pt.save_model(single_check, f"models//model-{model + 1}.pth")
    pt.save_model_results(f"models//model-{model + 1}-results.json")
    pt.reset_pt()