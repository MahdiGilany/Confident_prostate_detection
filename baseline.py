import time
import math
from collections import namedtuple

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from utils.misc import load_pkl
from utils.dataset import RandomSelect, ToTensor, RFDataset


class TransformerModel(nn.Module):
    def __init__(self, n_class, n_inp, n_head, n_hid, n_layers, dropout):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(n_inp, dropout)
        encoder_layers = TransformerEncoderLayer(n_inp, n_head, n_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.decoder = nn.Linear(n_inp, n_class)

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        init_range = .1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self, src):
        src = self.pos_encoder(src)  # Positional embedding
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output


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


def get_device(device_str=None):
    if device_str:
        try:
            torch.device('cuda')
        except Exception as E:
            raise E
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_preparation():
    data = load_pkl()
    composed_train = transforms.Compose([RandomSelect(), ToTensor()])
    rf_train = RFDataset(data['train'], transform=composed_train)
    data_train = DataLoader(rf_train, batch_size=128, shuffle=True, num_workers=0)

    composed_val = transforms.Compose([RandomSelect(1000), ToTensor()])

    rf_val = RFDataset(data['val'], transform=composed_val)
    data_val = DataLoader(rf_val, batch_size=1, shuffle=False, num_workers=0)

    rf_test = RFDataset(data['test'], transform=composed_val)
    data_test = DataLoader(rf_test, batch_size=1, shuffle=False, num_workers=0)

    return data_train, data_val, data_test


def get_model():
    """

    :return:
    """
    params = namedtuple('HyperParameters', [])
    params.n_class = 2
    params.em_size = 100  # embedding dimension
    params.n_hid = 100  # the dimension of the feedforward network model in nn.TransformerEncoder
    params.n_layers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
    params.n_head = 2  # the number of heads in the multi_head_attention models
    params.dropout = .0  # the dropout value
    model = TransformerModel(params.n_class, params.em_size, params.n_hid, params.n_head,
                             params.n_layers, params.dropout).to(get_device())
    return model, params


def get_trainer(model: TransformerModel):
    trainer = namedtuple('Trainer', ['criterion', 'optimizer', 'scheduler'])
    trainer.criterion = nn.CrossEntropyLoss()
    trainer.lr = 5.  # learning rate
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=trainer.lr)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1.0, gamma=.95)
    return trainer


def train(model: TransformerModel, n_class: int, data_train, epoch: int, trainer: namedtuple, device: torch.device):
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()

    for i, batch in enumerate(data_train):
        data, label = batch['data'].type(torch.float).to(device), \
                      batch['label'].type(torch.long).to(device)
        trainer.optimizer.zero_grad()
        output = model(data)
        loss = trainer.criterion(output.view(-1, n_class), label.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), .5)
        trainer.optimizer.step()

        total_loss += loss.item()
        log_interval = 2
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(epoch, i, len(data_train),
                                                      trainer.scheduler.get_last_lr()[0],
                                                      elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
    return model


def evaluate(eval_model: TransformerModel, n_class: int, data_source, trainer: namedtuple, device: torch.device):
    eval_model.eval()
    total_loss = 0.
    acc = 0.
    with torch.no_grad():
        for i, batch in enumerate(data_source):
            data, label = batch['data'].type(torch.float).to(device), \
                          batch['label'].type(torch.long).to(device)
            output = eval_model(data)
            output_flat, label = output.view(-1, n_class), label.view(-1)
            total_loss += len(data) * trainer.criterion(output_flat, label).item()
            core_acc = (F.softmax(output_flat, dim=1).argmax(dim=1) == label)
            core_acc = 1 if core_acc.sum() > len(core_acc) // 2 else 0
            acc += core_acc
            # print(F.softmax(output_flat, dim=1).argmax(dim=1))
        return total_loss / (len(data_source) - 1), 100 * acc / (len(data_source) - 1)


def main():
    device = get_device()

    train_data, val_data, test_data = data_preparation()

    model, params = get_model()

    trainer = get_trainer(model)
    best_val_loss = float('inf')
    epochs = 500  # The number of epochs
    best_model = None

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model = train(model, params.n_class, train_data, epoch, trainer, device)
        val_loss, val_acc = evaluate(model, params.n_class, val_data, trainer, device)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.4f}s | valid loss {:5.4f} | valid acc {:5.0f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, val_acc, math.exp(val_loss)))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        trainer.scheduler.step()


if __name__ == '__main__':
    main()
