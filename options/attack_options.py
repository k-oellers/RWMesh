import options


class AttackOptions(options.TrainOptions):
    def initialize(self):
        options.TrainOptions.initialize(self)
        self.parser.set_defaults(phase='attack')
        self.parser.add_argument("--gradient_weight", default=0.01, type=float)
        self.parser.add_argument("--max_attack_iterations", default=2000, type=int)
        self.parser.add_argument("--save_modified", action='store_true')
        self.parser.add_argument("--init_eval", action='store_true')
