import time
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchsummary import summary
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from utils.misc import load_pkl
from models.resnet1d_repo.resnet1d import ResNet1D
from models.resnet1d_ucr import ResNetBaseline
from models.inception1d import InceptionModel
from utils.sampler import BalancedBinaryRandomSampler
from utils.dataset import RandomSelect, ToTensor, RFDataset

N_TIMESERIES = 512
N_TIMESERIES_TEST = 3000
EPOCHS = 1000
N_TIMEPOINTS = 180
BATCH_SIZE = 2


def get_device(device_str=None):
    if device_str:
        try:
            torch.device('cuda')
        except Exception as E:
            raise E
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def data_preparation():
    data = load_pkl()
    composed_train = transforms.Compose([RandomSelect(N_TIMESERIES, N_TIMEPOINTS), ToTensor()])
    rf_train = RFDataset(data['train'], transform=composed_train)
    sampler_train = BalancedBinaryRandomSampler(rf_train, label=[x['label'][0] for x in rf_train])
    data_train = DataLoader(rf_train, batch_size=BATCH_SIZE, num_workers=0, sampler=sampler_train)

    composed_val = transforms.Compose([RandomSelect(N_TIMESERIES_TEST, N_TIMEPOINTS, n_separate_rows=1), ToTensor()])
    rf_val = RFDataset(data['val'], transform=composed_val)
    data_val = DataLoader(rf_val, batch_size=1, shuffle=False, num_workers=0)

    rf_test = RFDataset(data['test'], transform=composed_val)
    data_test = DataLoader(rf_test, batch_size=1, shuffle=False, num_workers=0)

    return data_train, data_val, data_test


def get_model(model_name):
    """

    :return:
    """
    params = namedtuple('HyperParameters', [])
    params.n_class = 2
    if model_name == 'transformer':
        from models.transformer import TransformerModel

        params.em_size = N_TIMEPOINTS  # embedding dimension
        params.n_hid_0 = 256  # the dimension of the feedforward network model in nn.TransformerEncoder
        params.n_hid_1 = 256  # the dimension of the feedforward network model in nn.TransformerEncoder
        params.n_layers = 2  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        params.n_head = 2  # the number of heads in the multi_head_attention models
        params.dropout = .0  # the dropout value
        model = TransformerModel(params.n_class, params.em_size, params.n_hid_0, params.n_hid_1, params.n_head,
                                 params.n_layers, params.dropout).to(get_device())
    if model_name == 'resnet_ucr':
        model = ResNetBaseline(in_channels=1, num_pred_classes=params.n_class).to(get_device())
    elif model_name == 'resnet1d':
        kernel_size = 16
        stride = 2
        n_block = 48
        downsample_gap = 6
        increasefilter_gap = 12
        params.n_class = 2
        model = ResNet1D(
            in_channels=1,
            base_filters=32,  # 64 for ResNet1D, 352 for ResNeXt1D
            kernel_size=kernel_size,
            stride=stride,
            groups=32,
            n_block=n_block,
            n_classes=params.n_class,
            downsample_gap=downsample_gap,
            increasefilter_gap=increasefilter_gap,
            use_do=True)
        model.to(get_device())
    if model_name == 'inception':
        model = InceptionModel(num_blocks=1, in_channels=1, out_channels=2,
                               bottleneck_channels=2, kernel_sizes=41, use_residuals=True,
                               num_pred_classes=params.n_class).to(get_device())

    return model, params


def get_trainer(model: nn.Module):
    trainer = namedtuple('Trainer', ['criterion', 'optimizer', 'scheduler'])
    trainer.criterion = nn.CrossEntropyLoss()
    # trainer.lr = 5.  # learning rate
    # trainer.optimizer = torch.optim.SGD(model.parameters(), lr=trainer.lr, weight_decay=0)
    # trainer.scheduler = torch.optim.lr_scheduler.StepLR(trainer.optimizer, 1.0, gamma=.95)
    trainer.lr = 1e-2  # learning rate
    trainer.optimizer = torch.optim.Adam(model.parameters(), lr=trainer.lr)  # , weight_decay=1e-3)
    # trainer.optimizer = torch.optim.SGD(model.parameters(), lr=trainer.lr)  # , weight_decay=1e-3)
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(trainer.optimizer,
                                                                   mode='min', factor=.1, patience=20)

    return trainer


def train(model: nn.Module, n_class: int, data_train, epoch: int, trainer: namedtuple, device: torch.device):
    model.train()  # Turn on the train mode
    total_loss = 0.

    start_time = time.time()
    log_interval = 50
    correct = 0
    n = 0

    for i, batch in enumerate(data_train):
        data, label = batch['data'].type(torch.float).to(device), \
                      batch['label'].type(torch.long).to(device)
        data = data.view(-1, 1, data.size(2))
        output = model(data)
        loss = trainer.criterion(output.view(-1, n_class), label.view(-1))
        trainer.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), .5)
        trainer.optimizer.step()

        total_loss += loss.item()
        output_flat, label = output.view(-1, n_class), label.view(-1)
        pred = F.softmax(output_flat, dim=1).argmax(dim=1)
        correct += (pred == label).sum()
        n += len(label)

        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.5f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | acc {:5f}'.format(epoch, i, len(data_train),
                                                    trainer.optimizer.param_groups[0]['lr'],
                                                    elapsed * 1000 / log_interval, cur_loss, correct / n))
            total_loss = 0
            start_time = time.time()
        # print(pred)
    return model


def evaluate(eval_model: nn.Module, n_class: int, data_source, trainer: namedtuple, device: torch.device):
    eval_model.eval()
    total_loss = 0.
    acc = 0.
    preds = []
    correct = 0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(data_source):
            data, label = batch['data'].type(torch.float).to(device), \
                          batch['label'].type(torch.long).to(device)
            data = data.view(-1, 1, data.size(2))
            output = eval_model(data)
            output_flat, label = output.view(-1, n_class), label.view(-1)
            total_loss += len(data) * trainer.criterion(output_flat, label).item()
            pred = F.softmax(output_flat, dim=1).argmax(dim=1)
            # core_acc = 1 if pred.sum() > (len(pred) // 4) else 0
            # acc += (core_acc == label[0])
            # # preds.append(pred.cpu().data.numpy()[0])
            # preds.append(pred.sum().cpu().item() / data.shape[0])

            correct += (pred == label).sum()
            n += len(label)
            # print(correct/n)
        # print(preds)
        return total_loss / (len(data_source) - 1), 100 * (correct / n)  # 100 * acc / (len(data_source) - 1)


def main():
    device = get_device()

    train_data, val_data, test_data = data_preparation()

    model, params = get_model('resnet1d')

    trainer = get_trainer(model)
    best_val_loss = float('inf')
    epochs = EPOCHS  # 60  # The number of epochs
    best_model = None
    to_eval = True
    # to_eval = False
    summary(model, (1, N_TIMEPOINTS), device='cuda')

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model = train(model, params.n_class, train_data, epoch, trainer, device)

        if to_eval:
            val_loss, val_acc = evaluate(model, params.n_class, val_data, trainer, device)
            print('-' * 80)
            print('| end of epoch {:3d} | time: {:5.4f}s | valid loss {:5.4f} | valid acc {:5.0f} | '
                  .format(epoch, (time.time() - epoch_start_time), val_loss, val_acc))
            print('-' * 80)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model

        trainer.scheduler.step(epoch - 1)


if __name__ == '__main__':
    main()
