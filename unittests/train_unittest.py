import unittest
import options
import util
import config


class Train(unittest.TestCase):
    def init_trainer(self, dataset, objective, custom_config=None) -> util.Trainer:
        config = util.get_default_configs()[dataset][objective]
        if custom_config:
            config.update(custom_config)
        opt = options.TrainOptions().parse(config)
        return util.Trainer(opt)

    def test_a_train_shrec16(self):
        trainer = self.init_trainer('shrec16', 'supervised', {'epochs': 1, 'name': 'unittest_supervised'})
        acc = trainer.train()
        self.assertTrue(acc > 0.08)

    def test_b_run_shrec16(self):
        trainer = self.init_trainer('shrec16', 'supervised', {'name': 'unittest_supervised'})
        acc = trainer.train()
        print(acc)
        self.assertTrue(acc > 0.033)

    def test_c_resume_shrec16(self):
        trainer = self.init_trainer('shrec16', 'supervised',
                                    {'epochs': 1, 'name': 'unittest_supervised', 'resume': True})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)

    def test_d_a_pretrain_shrec16(self):
        trainer = self.init_trainer('shrec16', 'barlow', {'epochs': 1, 'name': 'unittest_barlow'})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)

    def test_d_b_resume_pretrain_shrec16(self):
        trainer = self.init_trainer('shrec16', 'barlow', {'epochs': 1, 'name': 'unittest_barlow', 'resume': True})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)

    def test_d_c_test_pretrained_shrec16(self):
        trainer = self.init_trainer('shrec16', 'barlow', {'name': 'unittest_barlow', 'pretrained': True})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)

    def test_d_d_finetune_shrec16(self):
        trainer = self.init_trainer('shrec16', 'barlow',
                                    {'epochs': 1, 'name': 'unittest_barlow', 'finetune': 1, 'pretrained': True})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)

    def test_e_a_pretrain_finetune_shrec16(self):
        trainer = self.init_trainer('shrec16', 'barlow',
                                    {'epochs': 1, 'name': 'unittest_finetune_barlow', 'finetune': 1})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)

    def test_e_b_resume_finetune_pretrain_shrec16(self):
        trainer = self.init_trainer('shrec16', 'barlow',
                                    {'epochs': 1, 'name': 'unittest_finetune_barlow', 'finetune': 1, 'resume': True})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)

    def test_e_c_test_finetune_pretrained_shrec16(self):
        trainer = self.init_trainer('shrec16', 'barlow', {'name': 'unittest_finetune_barlow', 'pretrained': True})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)

    def test_e_e_finetune_finetuned_shrec16(self):
        trainer = self.init_trainer('shrec16', 'barlow',
                                    {'epochs': 1, 'name': 'unittest_finetune_barlow', 'finetune': 1,
                                     'pretrained': True})
        acc = trainer.train()
        self.assertTrue(acc > 0.033)


if __name__ == '__main__':
    unittest.main()
