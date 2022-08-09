import options
import util

if __name__ == '__main__':
    util.set_global_seed(0)
    opt = options.TestOptions().parse()
    util.load_opt(opt)
    # test
    accuracy = util.Trainer(opt).test()
    print(accuracy)
