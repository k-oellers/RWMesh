from .base_options import BaseOptions
from matplotlib.colors import Normalize, LogNorm

normalizer_options = {"log": LogNorm, "norm": Normalize}


class VisualizeOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument("--color_map", default="inferno")
        self.parser.add_argument("--normalizer", default="log", choices=normalizer_options.keys())
        self.parser.add_argument("--amount", default=1, type=int)
        self.parser.add_argument("--store_path", default="visualization")
        self.parser.add_argument('--phase', type=str, default='test', help='train, test')
        self.parser.add_argument("--visualize", default=True, help='return attention weights')
