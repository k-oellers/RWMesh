import util
import options

if __name__ == '__main__':
    opt = options.VisualizeOptions().parse()
    util.Trainer(opt).visualize()
