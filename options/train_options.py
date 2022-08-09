import options


class TrainOptions(options.BaseOptions):
    def initialize(self):
        options.BaseOptions.initialize(self)
        self.parser.add_argument("--epochs", default=10, type=int, help='number of training epochs')
        self.parser.add_argument("--learning_rate", default=2e-4, type=float, help='learning rate while training')
        self.parser.add_argument("--early_stop", default=8, type=int,
                                 help='stop training after n epochs of not increasing accuracy')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument("--optimizer", default="default", help='optimizer')
        # augmentation args
        self.parser.add_argument("--edge_drop", default=0.0, type=float, help='probability to apply edge drop')
        self.parser.add_argument("--edge_drop_deg", default=0.0, type=float,
                                 help='probability to apply edge drop (centrality based)')
        self.parser.add_argument("--edge_add", default=0.0, type=float, help='probability to apply edge drop')
        self.parser.add_argument("--noise", default=0.0, type=float, help='probability to apply noise')
        self.parser.add_argument("--noise_deg", default=0.0, type=float, help='probability to apply (centrality based)')
        self.parser.add_argument('--noise_weight', default=1e-3, type=float, help='noise weight')
        self.parser.add_argument("--visualize", default=False, help='return attention weights')
