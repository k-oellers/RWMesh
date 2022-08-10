import options
import util
from tqdm import tqdm
import os

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
    modified_meshes = trainer.attack()
    acc = trainer.test(trainer.model)
    print(f'model acc with modified meshes: {acc}')
    if opt.save_modified and modified_meshes:
        for seq in tqdm(trainer.test_set):
            for mesh in seq['mesh']:
                if mesh.filename in modified_meshes:
                    util.save_obj(mesh, os.path.join(util.get_vis_path(opt), 'modified_meshes', mesh.filename))
