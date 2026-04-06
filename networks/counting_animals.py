import torch
from torch import nn


class AnimalCounter(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.convolutional_layers = nn.Sequential(nn.Conv2d(3, 64, (10, 10), (4, 6)), 
                                                  nn.ReLU(), 
                                                  nn.Conv2d(64, 64, (6, 5), (3, 3)), 
                                                  nn.ReLU(), 
                                                  nn.Conv2d(64, 64, (3, 3), (2, 2)),
                                                  nn.ReLU())
        
        pos_range = torch.arange(0, 64)[..., None]
        dim_range = torch.arange(0, 780)

        pair_index = dim_range // 2
        angle = pos_range / (10000 ** (2 * pair_index / 780))

        even_mask = dim_range % 2 == 0
        odd_mask = dim_range % 2 == 1

        angle[:, even_mask] = torch.sin(angle[:, even_mask])
        angle[:, odd_mask] = torch.cos(angle[:, odd_mask])
        self.pos_enc = angle

        self.encoder = nn.TransformerEncoderLayer(780, 10, batch_first=True)
        self.encoder_layers = nn.TransformerEncoder(self.encoder, 8)

        self.feed_forward_layers = nn.Sequential(nn.Linear(780, 100),
                                                 nn.ReLU(), 
                                                 nn.Linear(100, 1), 
                                                 nn.Sigmoid())
        
    def forward(self, x):

        if len(x.shape) == 3:
            x = x.view(-1, *x.size())
        
        x = self.convolutional_layers(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = x + self.pos_enc
        x = self.encoder_layers(x)

        x = x[:, 0, :]
        x = self.feed_forward_layers(x)
        return x