import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerModel(nn.Module):
    def __init__(self, n_class, n_inp, n_hid_0, n_hid_1, n_head, n_layers, dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.encoder = nn.Linear(n_inp, n_hid_0)
        self.pos_encoder = PositionalEncoding(n_hid_0, dropout)
        encoder_layers = TransformerEncoderLayer(n_hid_0, n_head, n_hid_1, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(n_hid_1, n_class)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = .1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)  # Positional embedding
        output = self.transformer_encoder(src)
        output = output.mean(axis=1)
        return self.decoder(output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)