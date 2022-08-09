import argparse
import sys
import util
import data
from torch import cuda
from datetime import datetime
import torch
import models

position_modes = ['absolute', 'relative']
model_map = {"transformer": models.TransformerModel, "rnn": models.RNNModel}
supervised_map = {'barlow': models.BarlowTwins}
signatures = {'hks': data.compute_hks_autoscale, 'wks': data.compute_wks}


def init_opt(opt):
    # self-supervised objective must be given when finetuning or saving correlation matrix
    if opt.save_correlation:
        assert opt.self_supervised, 'save correlation only compatible with barlow twin model'

    opt.start_time = datetime.now().strftime('%m-%d-%Y_%H-%M-%S')
    # automatically set dataset mode
    opt.dataset_mode = data.get_dataset_class(opt.dataroot)
    if opt.dataset_mode is None:
        raise ValueError(f'unknown dataset type at path {opt.dataroot}')

    single_gpu = opt.model == 'rnn' or opt.single_gpu
    opt.device_ids = get_device_ids(single_gpu)
    opt.device = torch.device('cuda' if opt.device_ids else 'cpu')


def load_default_config(opt: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    arg_keys = [x[2:] for x in sys.argv[1:] if x.startswith('--')]
    train_objective = opt.self_supervised if opt.self_supervised else 'supervised'
    dataset_name = '_'.join([x.lower() for x in opt.dataroot.split('/')[1:]])
    model_default = util.get_default_configs()[dataset_name][train_objective]
    for key, value in model_default.items():
        if key in opt and key not in arg_keys and getattr(opt, key) == parser.get_default(key):
            setattr(opt, key, value)


def get_device_ids(single_gpu):
    if cuda.is_available():
        # use single gpu when using rnn or visualization
        return list(range(cuda.device_count())) if not single_gpu else [0]
    return None


class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False
        self.opt = None

    def initialize(self):
        # data args
        self.parser.add_argument('--dataroot', default='datasets/Cubes', help='path to meshes dataset')
        self.parser.add_argument('--default_config', action='store_true',
                                 help='use default configs from folder configs/dataset/training_objective')
        self.parser.add_argument('--cache_root', default='cache', help='save path for the cache folder')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--num_threads', default=3, type=int, help='number of threads for loading data')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples per epoch')
        self.parser.add_argument("--remesh", nargs='+', type=float, default=[],
                                 help='(multiple) remesh with n percent of meshes (example 0.5, 0.4)')
        # basic args
        self.parser.add_argument('--name', type=str, default='default', help='save and load name')
        self.parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        self.parser.add_argument("--walks", default=1, type=int, help='number of random walks per mesh')
        self.parser.add_argument("--sequence_length", type=int, default=100, help='length of the random walk')
        # model args
        self.parser.add_argument("--model", choices=model_map.keys(), default="transformer",
                                 help=f'used encoder model [{",".join(model_map.keys())}]')
        self.parser.add_argument("--position_mode", choices=position_modes, default="absolute",
                                 help='use relative or absolute position mode')
        self.parser.add_argument("--self_supervised", default=None, choices=supervised_map.keys(),
                                 help=f'use self-supervised training methods [{",".join(supervised_map.keys())}]')
        self.parser.add_argument("--resume", action='store_true', help='resume training')
        self.parser.add_argument("--single_gpu", action='store_true', help='force to train single-gpu')
        self.parser.add_argument("--no_rotation", action='store_false', help='deactivate rotation augmentation')
        # self-supervised args
        self.parser.add_argument("--embedding_size", default=64, type=int, help='embedding dimension of barlow twins')
        self.parser.add_argument("--temperature", default=0.001, type=float, help='temperature of barlow twins')
        self.parser.add_argument('--save_correlation', action='store_true',
                                 help='save the cross correlation matrix (just ssl learning)')
        self.parser.add_argument('--pretrained', action='store_true', help='finetune a pretrained model')
        self.parser.add_argument("--finetune", default=0, type=float,
                                 help='%% of models used for finetuning (only ssl)')
        self.parser.add_argument("--finetune_lr", default=1e-4, type=float,
                                 help='learning rate for the online finetuner')
        # testing
        self.parser.add_argument("--frequency", default=5, type=int,
                                 help='determines after how many trainings epochs the test and optional online finetuning is executed')
        self.parser.add_argument("--regex", default=None, help='regular expression to filter filenames')
        self.parser.add_argument("--log", action='store_true', help='activate tensorboard logging')
        # signature args
        self.parser.add_argument('--signature', choices=signatures, default='hks',
                                 help='used signature/spectral descriptor')
        self.parser.add_argument('--eig_basis', default=0, type=int,
                                 help='number of use eigenvalues and eigenvectors in laplace-beltrami operator')
        # transformer args
        self.parser.add_argument('--hidden_dim', type=int, default=256, help='hidden dim for transformer')
        self.parser.add_argument('--model_size', type=int, default=512, help='model size for transformer')
        self.parser.add_argument('--nlayers', type=int, default=8, help='number of transformer layers')
        self.parser.add_argument('--nheads', type=int, default=8, help='number of multi-attention heads')
        self.parser.add_argument("--pos_encoding", default=True, help='positional encoding')
        # directories
        self.parser.add_argument("--log_dir", type=str, default='run', help='tensorboard logging folder')
        self.parser.add_argument("--output_dir", type=str, default='output', help='output directory')
        self.parser.add_argument("--checkpoint_dir", type=str, default='store', help='checkpoint directory')
        self.parser.add_argument("--visualization_dir", type=str, default='visualization',
                                 help='visualization directory')
        self.initialized = True

    def parse(self, args=None):
        if type(args) == dict:
            args = util.config_to_args(args)
        # init parser
        if not self.initialized:
            self.initialize()
        # parse arguments
        self.opt, _ = self.parser.parse_known_args(args)
        # load default model
        if self.opt.default_config:
            load_default_config(self.opt, self.parser)
        init_opt(self.opt)
        return self.opt
