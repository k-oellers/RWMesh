from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import pathlib
import os
import copy
import options
import util


def train_transformer(config):
    opt = copy.copy(config['opt'])
    # update opt with current tuning parameters
    for key, value in config.items():
        if hasattr(opt, key):
            setattr(opt, key, value)

    # train
    util.Trainer(opt, report_tune=True).train()


def main(opt, num_samples=1, gpus_per_trial=1, metric='acc'):
    opt.dataroot = os.path.join(pathlib.Path().resolve(), opt.dataroot)
    opt.cache_root = os.path.join(pathlib.Path().resolve(), opt.cache_root)
    if opt.self_supervised:
        config = {
            # "nhid": tune.choice([512, 1024]),
            "nlayers": tune.grid_search([5, 7]),
            "model_size": tune.grid_search([256, 512, 1024, 2048]),
            # "hidden_dim": tune.choice([128, 256, 512, 1024]),
            # "embedding_size": tune.grid_search([16, 32, 64]),
            # "t": tune.loguniform(1e-2, 9e-6),
            "eig_basis": tune.grid_search([0, 8, 16]),
            "opt": opt,
            # "noise": tune.grid_search([0.0, 0.1]),
            # "noise_deg": tune.grid_search([0.0, 0.1]),
            "edge_drop": tune.grid_search([0, 0.25, 0.5]),
            "edge_drop_deg": tune.grid_search([0, 0.25, 0.5]),
            "edge_add": tune.grid_search([0, 0.25, 0.5]),
            "sequence_length": tune.grid_search([100, 200]),
            "noise_weight": tune.loguniform(1e-2, 1e-4),
        }
    else:
        config = {
            "nhid": tune.grid_search([256, 512, 1024]),
            "nlayers": tune.grid_search([4, 5, 6, 7]),
            "model_size": tune.grid_search([256, 512, 1024, 2048]),
            "hidden_dim": tune.grid_search([256, 512, 1024]),
            "sequence_length": tune.grid_search([100, 200, 252]),
            "eig_basis": tune.grid_search([0, 4, 8]),
            "opt": opt
        }
    if metric == 'loss':
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=opt.epochs,
            grace_period=2,
            reduction_factor=2)
    else:
        scheduler = ASHAScheduler(
            metric="accuracy",
            mode="max",
            max_t=opt.epochs,
            grace_period=2,
            reduction_factor=2)

    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(train_transformer),
        resources_per_trial={"gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        callbacks=[],
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    util.set_global_seed(0)
    main(options.TrainOptions().parse())
