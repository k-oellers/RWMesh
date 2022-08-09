import unittest
import options
import util
import os


class Train(unittest.TestCase):
    def test_a_train(self):
        name = 'test'
        config = {
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '.pt'))

    def test_a_train_pretrained(self):
        name = 'test'
        config = {
            'name': name,
            'pretrained': True
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '.pt'))

    def test_b_train_resume(self):
        name = 'test'
        config = {
            'resume': True,
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '.pt'))

    def test_b_train_resume_pretrained(self):
        name = 'test'
        config = {
            'resume': True,
            'pretrained': True,
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '.pt'))

    def test_c_test_train(self):
        name = 'test'
        config = {
            'phase': 'test',
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '.pt'))

    def test_d_pretrain(self):
        name = 'test'
        config = {
            'self_supervised': 'barlow',
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '.pt'))

    def test_e_finetune(self):
        name = 'test'
        config = {
            'self_supervised': 'barlow',
            'pretrained': True,
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '_finetuned' + '.pt'))

    def test_f_pretrain_resume(self):
        name = 'test'
        config = {
            'self_supervised': 'barlow',
            'resume': True,
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '.pt'))

    def test_g_finetune_resume(self):
        name = 'test'
        config = {
            'self_supervised': 'barlow',
            'resume': True,
            'pretrained': True,
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '_finetuned' + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '_finetuned' + '.pt'))

    def test_h_pretrain_test(self):
        name = 'test'
        config = {
            'self_supervised': 'barlow',
            'phase': 'test',
            'name': name
        }
        settings = options.TrainOptions().parse(config)
        self.assertTrue(settings.load_path.endswith(name + '_finetuned' + '.pt'))
        self.assertTrue(settings.save_path.endswith(name + '_finetuned' + '.pt'))


if __name__ == '__main__':
    unittest.main()
