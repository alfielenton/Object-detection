import torch
from torch import nn


class SingularityCheck(nn.Module):

    def __init__(self):

        super().__init__()
        
        self.convolutional_layers = nn.Sequential(nn.Conv2d(3, 64, (10, 10), (4, 6)), 
                                                  nn.ReLU(),
                                                  nn.Dropout(0.5), 
                                                  nn.Conv2d(64, 64, (6, 5), (3, 3)), 
                                                  nn.ReLU(),
                                                  nn.Dropout(0.5), 
                                                  nn.Conv2d(64, 64, (3, 3), (2, 2)),
                                                  nn.ReLU(),
                                                  nn.Dropout(0.5)) 
        
        self.final_conv = nn.Sequential(nn.Conv2d(64, 64, (3, 4), (3, 2)),
                                        nn.ReLU())

        self.feed_forward_layers = nn.Sequential(nn.Linear(7680, 2048),
                                                 nn.ReLU(),
                                                 nn.Dropout(), ##CHANGE TO DROPOUT?
                                                 nn.Linear(2048, 512),
                                                 nn.ReLU(),
                                                 nn.Dropout(), ##CHANGE TO DROPOUT?
                                                 nn.Linear(512, 1))
        
    def forward(self, x):

        batch_size = 1 if x.ndim == 3 else x.size(0)
        
        x = self.convolutional_layers(x)
        x = self.final_conv(x)
        
        x = x.view(batch_size, -1)

        x = self.feed_forward_layers(x)
        return x