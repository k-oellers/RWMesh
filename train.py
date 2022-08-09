import options
import util

if __name__ == '__main__':
    util.set_global_seed(0)
    opt = options.TrainOptions().parse()
    util.Trainer(opt).train_full()
