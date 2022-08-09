import options
import util

if __name__ == '__main__':
    util.set_global_seed(0)
    opt = options.AttackOptions().parse()
    trainer = util.Trainer(opt)
    if opt.init_eval:
        acc = trainer.test(trainer.model)
        imi_acc = trainer.test(trainer.imi_model)
        print(f'model acc: {acc}. imitated model acc: {imi_acc}')

    if not opt.pretrained:
        trainer.train()
    trainer.attack()
