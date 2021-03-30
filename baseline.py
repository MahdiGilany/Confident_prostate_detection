import time
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.data import DataLoader

from utils.misc import load_pkl
from utils.dataset import RandomSelect, ToTensor, RFDataset


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


def get_model(model_name):
    """

    :return:
    """
    if model_name == 'transformer':
        from models.transformer import TransformerModel
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


def get_trainer(model: nn.Module):
    trainer = namedtuple('Trainer', ['criterion', 'optimizer', 'scheduler'])
    trainer.criterion = nn.CrossEntropyLoss()
    trainer.lr = 5.  # learning rate
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=trainer.lr)
    trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1.0, gamma=.95)
    return trainer


def train(model: nn.Module, n_class: int, data_train, epoch: int, trainer: namedtuple, device: torch.device):
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


def evaluate(eval_model: nn.Module, n_class: int, data_source, trainer: namedtuple, device: torch.device):
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

    model, params = get_model('transformer')

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
