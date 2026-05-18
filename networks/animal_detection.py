import torch
from torch import nn
import time

class AnimalDetector(nn.Module):

    def __init__(self, num_classes, ydims, embedding_dims, num_encoder_layers, num_decoder_layers, num_attn_heads):

        super().__init__()

        self.num_classes = num_classes
        self.ydims = ydims
        self.embedding_dims = embedding_dims
        self.num_encoder_layers = num_encoder_layers
        self.num_obj_decoder_layers = num_decoder_layers // 2
        self.num_bc_decoder_layers = num_decoder_layers - self.num_obj_decoder_layers
        self.num_attn_heads = num_attn_heads

        assert self.embedding_dims % self.num_attn_heads == 0, "embedding dims must be divisible by number of attention heads"

        self.convolutional_base = nn.Sequential(nn.Conv2d(3, 96, 12, 4), 
                                                nn.ReLU(),
                                                nn.Dropout(0.1), 
                                                nn.Conv2d(96, 125, 6, 3), 
                                                nn.ReLU(), 
                                                nn.MaxPool2d(2, 1), 
                                                nn.Conv2d(125, 215, 3, 2), 
                                                nn.ReLU(),
                                                nn.Dropout(0.1), 
                                                nn.Conv2d(215, 215, 2, 1), 
                                                nn.ReLU())
        
        self.convolutional_fc_layers = nn.Sequential(nn.Linear(324, 1024),
                                                     nn.ReLU(),
                                                     nn.Dropout(0.1), 
                                                     nn.Linear(1024, self.embedding_dims))
        
        self.img_encoder_layer = nn.TransformerEncoderLayer(self.embedding_dims, self.num_attn_heads, batch_first=True)
        self.img_encoder = nn.TransformerEncoder(self.img_encoder_layer, self.num_encoder_layers)

        self.c_start_embedder = nn.Linear(1, self.embedding_dims)
        self.c_embedder = nn.Linear(self.num_classes, self.embedding_dims)

        self.b_start_embedder = nn.Linear(1, self.embedding_dims)
        self.b_embedder = nn.Linear(self.ydims, self.embedding_dims)

        self.object_decoder_layer = nn.TransformerDecoderLayer(self.embedding_dims, self.num_attn_heads, batch_first=True)
        self.object_decoder = nn.TransformerDecoder(self.object_decoder_layer, self.num_obj_decoder_layers)

        self.c_decoder_layer = nn.TransformerDecoderLayer(self.embedding_dims, self.num_attn_heads, batch_first = True)
        self.c_decoder = nn.TransformerDecoder(self.c_decoder_layer, self.num_bc_decoder_layers)

        self.b_decoder_layer = nn.TransformerDecoderLayer(self.embedding_dims, self.num_attn_heads, batch_first=True)
        self.b_decoder = nn.TransformerDecoder(self.b_decoder_layer, self.num_bc_decoder_layers)

        self.c_ffn = nn.Sequential(nn.Linear(self.embedding_dims, 1024),
                                   nn.Dropout(0.1),
                                   nn.Linear(1024, self.num_classes + 1),
                                   nn.LogSoftmax(dim=1))
        
        self.b_ffn = nn.Sequential(nn.Linear(self.embedding_dims, 1024),
                                   nn.Dropout(0.1),
                                   nn.Linear(1024, self.ydims),
                                   nn.ReLU())

    def find_max_seq_len(self, seqs):
        m = 0
        for s in seqs:
            m = len(s) if len(s) > m else m
        return m
    
    def generate_class_embeddings(self, c_seqs, device):

        c_start_embed = self.c_start_embedder(torch.tensor([[1.]]).to(device))
        max_seq_len = self.find_max_seq_len(c_seqs)

        batch = []
        padding_mask = []

        for seq in c_seqs:

            if seq:
                one_hots = []
                for c in seq:
                    oh = torch.zeros(self.num_classes)
                    oh[c] = 1.
                    one_hots.append(oh)
                
                one_hots = torch.stack(one_hots).to(device)
                c_embeds = self.c_embedder(one_hots)
                pad_len = max_seq_len - c_embeds.size(0)
                zero_pad = torch.zeros((pad_len, self.embedding_dims)).to(device)
                c_embeds = torch.cat([c_start_embed, c_embeds, zero_pad])

            else:
                pad_len = max_seq_len
                zero_pad = torch.zeros((pad_len, self.embedding_dims)).to(device)
                c_embeds = torch.cat([c_start_embed, zero_pad])
            
            batch.append(c_embeds)

            mask = [False] * (1 + len(seq)) + [True] * pad_len
            padding_mask.append(mask)

        
        return torch.stack(batch), torch.tensor(padding_mask).to(device)
    
    def generate_box_embedding(self, b_seqs, device):

        b_start_embed = self.b_start_embedder(torch.tensor([[1.]]).to(device))
        max_seq_len = self.find_max_seq_len(b_seqs)

        batch = []
        for seq in b_seqs:

            if seq:
                b_embeds = self.b_embedder(torch.tensor(seq).to(device))
                pad_len = max_seq_len - b_embeds.size(0)
                zero_pad = torch.zeros((pad_len, self.embedding_dims)).to(device)
                b_embeds = torch.cat([b_start_embed, b_embeds, zero_pad])

            else:
                zero_pad = torch.zeros((max_seq_len, self.embedding_dims)).to(device)
                b_embeds = torch.cat([b_start_embed, zero_pad])
                
            batch.append(b_embeds)
        
        return torch.stack(batch)

    def forward(self, X, c_seqs, b_seqs):

        c_embeddings, obj_padding_mask = self.generate_class_embeddings(c_seqs, X.device)
        b_embeddings = self.generate_box_embedding(b_seqs, X.device)
        obj_embedding = c_embeddings + b_embeddings

        X = self.convolutional_base(X)
        X = X.view(X.size(0), X.size(1), -1)
        X = self.convolutional_fc_layers(X)

        X = self.img_encoder(X)
        X_obj = self.object_decoder(obj_embedding, X, tgt_key_padding_mask = obj_padding_mask)

        b_dec = self.b_decoder(X_obj, X)[:, -1, :]
        c_dec = self.c_decoder(X_obj, X)[:, -1, :]

        b = self.b_ffn(b_dec)
        c = self.c_ffn(c_dec)

        return c, b