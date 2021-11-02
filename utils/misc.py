import io
import os
import sys
import pickle
import random

sys.path.insert(0, '../..')

import cv2
import yaml
import time
import argparse
import functools

import torch
import numpy as np
from munch import munchify
from yaml import CLoader as Loader
from tensorboardX import SummaryWriter

import bz2
import _pickle as cPickle

import mat73
from scipy.io import matlab

from datetime import datetime

import numpy as np


def timer(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        if 'skip_timer' in kwargs.keys():
            skip_timer = kwargs['skip_timer']
            kwargs.pop('skip_timer')
            if skip_timer:
                return func(*args, **kwargs)

        start_time = time.perf_counter()  # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()  # 2
        run_time = end_time - start_time  # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value

    return wrapper_timer


class Logger(object):
    def __init__(self, filename, directory='./../logs'):
        self.terminal = sys.stdout
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.log = open(f"{directory}/{filename}.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def print_date_time():
    # datetime object containing current date and time
    now = datetime.now()

    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f'\n\n{("* " * 20)}{dt_string}{(" *" * 20)}')


def norm_01(x: np.ndarray):
    return (x - x.min()) / x.max()


def crop_bbox(x, bbox):
    """

    :param x: size = [..., H, W]
    :param bbox: Bounding box (min_row, min_col, max_row, max_col)
    :return:
    """
    if x is None:
        return x
    return x[..., bbox[0]:bbox[2], bbox[1]:bbox[3]]


def paste_bbox(x, original_shape, bbox):
    """

    :param x: the cropped image
    :param original_shape: shape of the original rf image
    :param bbox: output of regionprops
    :return:
    """
    if x is None:
        return x
    x_large = np.zeros(original_shape, dtype=x.dtype)
    x_large[:, bbox[0]:bbox[2], bbox[1]:bbox[3]] = x
    return x_large


def load_pickle(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def save_pickle(obj, filename):
    with open(filename, 'wb') as fp:
        pickle.dump(obj, fp)


def compressed_pickle(title, data):
    """Takes much longer time to save file, but much more storage-friendly"""
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)  # Load any compressed pickle file


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = cPickle.load(data)
    return data


def parse_args() -> dict:
    """Read commandline arguments
    Argument list includes mostly tunable hyper-parameters (learning rate, number of epochs, etc).
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path for the config file")
    parser.add_argument("--backbone", type=str,
                        help="Backbone name")
    parser.add_argument("--exp-name", type=str,
                        help="Experiment name")
    parser.add_argument("--loss-name", type=str,
                        help="Name of the loss function use with coteaching")
    parser.add_argument("--exp-suffix", type=str,
                        help="Suffix in the experiment name")
    parser.add_argument("--num-workers", type=int,
                        help="Training seed")
    parser.add_argument("--seed", type=int,
                        help="Training seed")
    parser.add_argument("--train-batch-size", type=int,
                        help="Batch size during training")
    parser.add_argument("--test-batch-size", type=int,
                        help="Batch size during test or validation")
    parser.add_argument("--n-epochs", type=int,
                        help="Batch size during training")
    parser.add_argument("--total-iters", type=int,
                        help="Batch size during training")
    parser.add_argument("--lr", type=float,
                        help="Learning rate")
    parser.add_argument("--min-inv", type=float,
                        help="Minimum involvement for cancer cores in training set")
    parser.add_argument("--elr_alpha", type=float,
                        help="lambda in early-learning regularization")
    parser.add_argument("--gpus-id", type=int, nargs='+', default=[0],
                        help="Path for the config file")
    parser.add_argument("--eval", action='store_true', default=False,
                        help='Perform evaluation; Training is performed if not set')
    parser.add_argument("--core_th", type=float, default=0.5,
                        help='Threshold used to convert signals predictions to cores')
    args = parser.parse_args()

    # Remove arguments that were not set and do not have default values
    args = {k: v for k, v in args.__dict__.items() if v is not None}
    return args


def read_yaml(verbose=False, setup_dir=False) -> yaml:
    """Read config files stored in yml"""
    # Read commandline arguments
    args = parse_args()
    if verbose:
        print_separator('READ YAML')

    # Read in yaml
    with open(args['config']) as f:
        opt = yaml.load(f, Loader)
    # Update option with commandline argument
    opt.update(args)
    # Convert dictionary to class-like object (just for usage convenience)
    opt = munchify(opt)

    if setup_dir:
        opt = setup_directories(opt)

    if verbose:
        # print yaml on the screen
        lines = print_yaml(opt)
        for line in lines:
            print(line)

    return opt


def fix_random_seed(seed):
    """Ensure reproducible results"""
    import torch
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def print_separator(text, total_len=50):
    print('#' * total_len)
    left_width = (total_len - len(text)) // 2
    right_width = total_len - len(text) - left_width
    print("#" * left_width + text + "#" * right_width)
    print('#' * total_len)


def print_yaml(opt):
    lines = []
    if isinstance(opt, dict):
        for key in opt.keys():
            tmp_lines = print_yaml(opt[key])
            tmp_lines = ["%s.%s" % (key, line) for line in tmp_lines]
            lines += tmp_lines
    else:
        lines = [": " + str(opt)]
    return lines


def create_path(opt):
    for k, v in opt['paths'].items():
        if not os.path.isfile(v):
            os.makedirs(v, exist_ok=True)


def setup_directories(opt):
    opt.exp_suffix = '' if opt.exp_suffix == 'none' else opt.exp_suffix
    if opt.paths is None:
        raise TypeError('log_dir, result_dir, and checkpoint_dir need to be specified!')
    prefix = '_'.join(opt.backbone) if isinstance(opt.backbone, list) else opt.backbone
    for k, v in opt.paths.__dict__.items():
        opt.paths[k] = v.replace('exp_name',
                                 '/'.join([opt.exp_name, prefix + opt.exp_suffix]))
    # commented to run on server
    if not ('home/' in opt.project_root):
        opt.data_source.data_root = '/'.join((opt.project_root, opt.data_source.data_root))
    if hasattr(opt.paths, 'self_train_checkpoint'):
        opt.paths.self_train_checkpoint = '/'.join((opt.project_root, opt.paths.self_train_checkpoint))

    # make directories
    create_path(opt)
    return opt


def get_time_suffix() -> str:
    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("_%Y%m%d_%H%M%S")
    return dt_string


def print_date_time():
    from datetime import datetime
    # datetime object containing current date and time
    now = datetime.now()
    # dd/mm/YY H:M:S
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    print(f'\n\n{("* " * 20)}{dt_string}{(" *" * 20)}')


def main():
    get_time_suffix()


class Logger(object):
    def __init__(self, filename, directory='./../logs'):
        self.terminal = sys.stdout
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.log = open(f"{directory}/{filename}.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def setup_tensorboard(opt):
    """

    :param opt: return from read_yaml
    :return:
    """
    writer = SummaryWriter(
        logdir=opt.paths.log_dir,
        flush_secs=opt.tensorboard.flush_secs,
        filename_suffix=opt.tensorboard.filename_suffix
    )
    return writer


def eval_mode(func):
    """wrapper for torch evaluation"""

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    return wrapper


def plot_to_image(fig, dpi=200):
    """Convert figure to image"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def init_weights(net, init_fn):
    from torch import nn

    # out_counter = 0
    for child in net.children():
        if isinstance(child, nn.Conv1d):
            init_fn(child.weights)


########################################################################
def load_matlab(filename, dynmc=False):
    # import h5py
    # with h5py.File(filename, 'rb') as f:
    #     return f.keys()

    if not dynmc:
        try:
            return mat73.loadmat(filename)
        except:
            with open(filename, 'rb') as fp:
                return matlab.loadmat(fp, struct_as_record=False, squeeze_me=True)
    else:
        with open(filename, 'rb') as fp:
            # try:
                return matlab.loadmat(fp, struct_as_record=False, squeeze_me=True)
            # except:
            #     print(filename)
            #     sys.exit()

def squeeze_Exact(inputdata):
    """Squeeze all data in Exact"""
    inputdata["data_train"] = inputdata["data_train"][0]
    inputdata["label_train"] = inputdata["label_train"][0]
    inputdata["data_val"] = inputdata["data_val"][0]
    inputdata["label_val"] = inputdata["label_val"][0]
    inputdata["data_test"] = inputdata["data_test"][0]
    inputdata["label_test"] = inputdata["label_test"][0]

