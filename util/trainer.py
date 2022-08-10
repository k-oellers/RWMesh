import argparse

from torch.optim import Optimizer

import options
import data
import util
import models
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from ray import tune
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os
import copy
from typing import Dict, Tuple, List, Optional, Union


class Trainer:
    """
    A unified class for training, testing, visualizing and attacking the mesh-based approach.
    """

    def __init__(self, opt: argparse.Namespace, report_tune: bool = False) -> None:
        """
        Initializes the datasets, model, tensorboard writer and loss.

        :param opt: input arguments
        :param report_tune: if True, report loss and accuracy with ray tune
        """

        self.opt = opt
        self.report_tune = report_tune

        # load datasets
        self.train_set = data.get_train_dataset(opt) if opt.phase != 'test' else None
        self.test_set = data.get_test_dataset(opt)
        self.finetune_set = data.get_finetune_dataset(opt) if opt.self_supervised and opt.pretrained else None

        # load tensorboard writer
        self.writer = self.get_writer()
        # copy number of classes to options namespace
        self.opt.n_classes = self.test_set.dataset.n_classes
        # load model, optimizer, lr_scheduler and optional start epoch
        self.model, self.optimizer, self.lr_scheduler, self.start_epoch = models.get_model(opt)
        # init loss criterion
        self.criterion = self.get_criterion()
        self.pretrain = self.opt.self_supervised and not self.opt.pretrained
        # init optional online finetuner
        self.finetuner, self.finetuner_opt = self.init_finetuner()
        if opt.phase == 'attack':
            # init imitation model when attacking
            self.imi_model, self.optimizer, self.lr_scheduler, _ = models.get_model(self.opt, True)
        else:
            self.imi_model = None

    def get_sequence_data(self, seq: Dict[str, np.ndarray]) -> Tuple[
        torch.Tensor, Union[torch.Tensor, int], torch.Tensor, torch.Tensor]:
        """
        Converts a dictionary of numpy arrays to a list of torch tensors

        :param seq: dictionary with keys 'walks', 'labels', 'pads', 'perm' with a corresponding numpy array
        :return: tensors of walks, labels, pads and perm
        """

        walks, labels, pads, perm = [
            torch.from_numpy(seq[arr]).to(self.opt.device) if type(seq[arr]) == np.ndarray else seq[arr] for arr in
            ['walks', 'labels', 'pads', 'perm']]
        labels = labels.long() if type(labels) == torch.Tensor else labels
        return walks, labels, pads, perm

    @staticmethod
    def stack_labels(labels: torch.Tensor, walks: int) -> torch.Tensor:
        return torch.stack((labels,) * walks).T.reshape(-1)

    def train_full(self):
        """
        Trains the model. After the training is done and the model is self-supervised, the model is also finetuned.

        :return: accuracy of the model
        """

        acc = self.train()
        if self.opt.self_supervised and not self.opt.pretrained:
            self.opt.pretrained = True
            acc = self.train()
        return acc

    def train(self) -> float:
        """
        Trains the model for a maximum amount of epochs. After opt.frequency epochs, the accuracy of the model for
        the train and test set is evaluated. Optionally, the training is stopped if the test accuracy does not increase
        after opt.early_stop test cycles. If self-supervised learning is the training objective, the model is
        automatically (online) finetuned with the train dataset with every test cycle. If opt.save_correlation, the
        cross correlation matrix of the barlow twins is saved with every test cycle. If the phase is attack, load an
        identical model and train this model to imitate the original results.

        :return: the accuracy of the model
        """

        early_stop_counter = best_accuracy = 0

        # use tqdm when not tuning
        rng = self.get_rng()

        # loop over epochs
        for epoch_id in rng:
            # optionally update tqdm with current phase
            self.update_tqdm(rng, 'training')
            loss_sum = train_size = 0
            # main training loop
            for step, seq in enumerate(self.train_set, start=epoch_id * len(self.train_set)):
                inputs, labels, pads, perm = self.get_sequence_data(seq)
                self.optimizer.zero_grad()
                if self.imi_model is not None:
                    # call imitation model
                    outputs = F.log_softmax(self.imi_model(inputs, pads), dim=1)
                    with torch.no_grad():
                        # labels are the softmax prob of the original model's output
                        labels = F.softmax(self.model(inputs, pads), dim=1)
                else:
                    # call model
                    outputs = self.model(inputs, pads)
                # calc loss
                loss = self.calc_loss(outputs, labels)
                loss_sum += loss.item()
                train_size += inputs.shape[0]
                # backward propagation and optimizer update
                self.update_optimizer(loss, self.optimizer)

            # adjust learning rate scheduler
            self.adjust_learning_rate()
            # log loss
            loss_sum = loss_sum / train_size
            # optionally report loss to ray tune and tensorboard
            self.update_writer("Loss/epoch", loss_sum, epoch_id)
            self.report({'loss': loss_sum})
            # for every n epochs...
            if (epoch_id - self.start_epoch) % self.opt.frequency == 0:
                # save cross correlation matrix
                if self.opt.save_correlation and self.opt.self_supervised and not self.opt.pretrained and not self.report_tune:
                    self.print_cc(epoch_id)

                model = self.model if self.imi_model is None else self.imi_model
                # optionally update tqdm with current phase
                self.update_tqdm(rng, 'testing')
                # testing and optionally finetune
                train_acc, test_acc = self.test_datasets(model, epoch_id)
                early_stop_counter += 1
                # save model and reset early stop counter when new best test accuracy is achieved
                if test_acc > best_accuracy:
                    self.update_tqdm(rng, f'test acc: {test_acc}', True)
                    best_accuracy = test_acc
                    early_stop_counter = 0
                    util.save_state(util.get_save_path(self.opt), model, epoch_id, self.opt, self.optimizer)
                # early stop reached
                elif early_stop_counter > self.opt.early_stop:
                    print('early stop')
                    return best_accuracy

        return best_accuracy

    def test_datasets(self, model, epoch_id) -> List[float]:
        """
        Tests the accuracy of the given model for the trainset and testsets.

        :param model: tested model
        :param epoch_id: current epoch
        :return: train and test accuracy
        """

        test_datasets = {'train': self.finetune_set if self.finetune_set else self.train_set, 'test': self.test_set}
        result = []
        for name, dataset in test_datasets.items():
            accuracy = self.test(model, dataset, name == 'train')
            self.update_writer(f"acc/{name}", accuracy, epoch_id)
            self.report({'accuracy': accuracy})
            result.append(accuracy)
        return result

    def adjust_learning_rate(self) -> None:
        """
        Adjust the learning rate of the scheduler if given
        """

        if self.lr_scheduler:
            self.lr_scheduler.step()

    def get_writer(self) -> Optional[SummaryWriter]:
        """
        Init the tensorboard writer

        :return: tensorboard writer if opt.log, else None
        """
        if not self.opt.log:
            return None

        writer = SummaryWriter(log_dir=util.get_log_path(self.opt))
        for key, value in vars(self.opt).items():
            writer.add_text(key, str(value), 0)
        return writer

    def get_rng(self) -> Union[tqdm, range]:
        """
        Returns the range, starting from start_epoch to start_epoch + epochs.

        :return: When ray tune is not used, return tqdm with given range, else simple python range object
        """
        rng = range(self.start_epoch, self.opt.epochs + self.start_epoch)
        if not self.report_tune:
            return tqdm(rng)
        return rng

    @staticmethod
    def update_tqdm(rng: Union[tqdm, range], desc: str, postfix: bool = False) -> None:
        """
        If rng is a tqdm, update the description or postfix with given text

        :param rng: optional tqdm object
        :param desc: used text
        :param postfix: if True, use postfix, else use description of tqdm
        :return:
        """

        if isinstance(rng, tqdm):
            if postfix:
                rng.set_postfix_str(desc)
            else:
                rng.set_description(desc)

    def report(self, metric: Dict[str, float]) -> None:
        """
        Report metrics to ray tune

        :param metric: dictionary of metrics
        :return:
        """
        if self.report_tune:
            tune.report(**metric)

    def update_writer(self, tag: str, scalar_value: Union[float, str], global_step: int) -> None:
        if self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def calc_loss(self, outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the loss for the different loss functions CrossEntropyLoss, BarlowTwinsLoss and KLDivLoss.

        :param outputs: output of the model
        :param labels: optional labels
        :return: single value tensor containing the loss
        """

        if isinstance(self.criterion, models.BarlowTwinsLoss):
            # self-supervised loss
            return self.criterion(*outputs)
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            batch_size, walks = outputs.shape[0], outputs.shape[1]
            if self.opt.dataset_mode == data.SegmentationData:
                labels = labels.reshape(-1, labels.shape[-1])
                outputs = outputs.reshape(batch_size * walks, outputs.shape[2], -1).permute(0, 2, 1)
            else:
                labels = self.stack_labels(labels, walks)
                outputs = outputs.reshape(batch_size * walks, -1)
            # cross entropy loss
            return self.criterion(outputs, labels)
        elif isinstance(self.criterion, nn.KLDivLoss):
            # cross entropy loss or KLDivLoss
            return self.criterion(outputs, labels)
        else:
            raise NotImplementedError('unknown loss criterion')

    def get_criterion(self) -> nn.Module:
        """
        Returns the loss function for the current objective: CrossEntropyLoss for supervised learning, BarlowTwinsLoss
        for self-supervised learning and KLDivLoss for imitation learning.

        :return: loss module
        """
        if self.opt.phase == 'attack':
            return torch.nn.KLDivLoss(reduction='batchmean')
        if self.opt.self_supervised and not self.opt.pretrained:
            return models.BarlowTwinsLoss(self.opt.temperature)
        return nn.CrossEntropyLoss()

    def init_finetuner(self) -> Tuple[Optional[nn.Linear], Optional[torch.optim.Optimizer]]:
        """
        Initialize the linear head and Adam optimizer for online finetuner

        :return: linear head and Adam optimizer
        """

        if not self.opt.self_supervised or self.opt.pretrained:
            return None, None

        finetuner = nn.Linear(self.model.encoder.fc.in_features, self.opt.n_classes).to(self.opt.device)
        finetuner_optimizer = torch.optim.Adam(finetuner.parameters(), lr=self.opt.finetune_lr)
        return finetuner, finetuner_optimizer

    def print_cc(self, epoch: Union[int, str]) -> None:
        """
        Generate the mean cross correlation matrix over all meshes in the testset. The result is shown and save under
        the path output/visualization/dataset/model/train_approach/name/cc_epoch.png

        :param epoch: current epoch (used for postfix)
        """
        with torch.no_grad():
            cc_matrices = []
            for seq in self.test_set:
                inputs, labels, pads, perm = self.get_sequence_data(seq)
                cc_matrix = self.model.cross_correlation_matrix(inputs, pads)
                cc_matrices.append(cc_matrix)

            cc_matrix = torch.mean(torch.stack(cc_matrices), dim=0)
            plt.imshow(cc_matrix.cpu().numpy())
            plt.savefig(f'{util.get_vis_path(self.opt)}{os.path.sep}cc_{epoch}.png')
            plt.show()

    def update_optimizer(self, loss: torch.Tensor, optimizer: Optimizer) -> None:
        """
        Back propagate loss and update Adam optimizer

        :param loss: loss tensor
        :param optimizer: optimizer object
        """
        loss.backward()
        optimizer.step()

    def test_segmentation(self, model: nn.Module, dataset: data.DataLoader, finetune: bool) -> float:
        """
        Calculates the accuracy of the segmentation task for a given model and dataset.
        It measures the mean of the correctly classified vertices. The vertex labels can be derived from the labels for
        the faces, which makes them ambiguous, since a vertex can be used by several faces with different labels.
        Instead, soft labels are used, which indicate the probability that the vertex belongs to a certain class.
        A vertex is considered correctly classified if the probability in the soft label for the predicted class is
        not zero.

        :param model: tested model
        :param dataset: tested dataset
        :param finetune: if True, finetune on the given dataset. Defaults to False
        :return: accuracy
        """

        correct = 0
        n_meshes = 0

        for seq in dataset:
            inputs, labels, pads, perm = self.get_sequence_data(seq)

            with torch.no_grad():
                outputs = model(inputs, pads)
            if self.finetuner:
                outputs = self.online_finetune(outputs, finetune)
                outputs = outputs.permute(0, 3, 1, 2)
                loss = F.cross_entropy(outputs, labels)
                if finetune:
                    self.update_optimizer(loss, self.finetuner_opt)
                outputs = outputs.permute(0, 2, 3, 1)
            # b, w, n, c
            outputs = torch.softmax(outputs, dim=1)
            for mesh_id, mesh in enumerate(seq['mesh']):
                av_pred = self.calc_average_prediction(perm[mesh_id], outputs[mesh_id], mesh)
                correct += self.segmentation_accuracy(mesh, av_pred)
                n_meshes += mesh.vertices.shape[0]
        return correct / n_meshes

    def test_classification(self, model: nn.Module, dataset: data.DataLoader, finetune: bool) -> float:
        """
        Measures the accuracy of the classification task for the given model and dataset.

        :param model: tested model
        :param dataset: tested dataset
        :param finetune: if True, finetune on the given dataset. Defaults to False
        :return: accuracy
        """
        correct = 0
        n_meshes = 0
        for seq in dataset:
            inputs, labels, pads, perm = self.get_sequence_data(seq)

            with torch.no_grad():
                outputs = model(inputs, pads)

            if self.finetuner:
                outputs = self.online_finetune(outputs, finetune)
                loss = F.cross_entropy(util.union_walks(outputs), self.stack_labels(labels, inputs.shape[1]))
                if finetune:
                    self.update_optimizer(loss, self.finetuner_opt)

            outputs = torch.softmax(outputs, dim=-1)
            correct += self.classification_accuracy(outputs, labels)
            n_meshes += labels.shape[0]
        return correct / n_meshes

    @staticmethod
    def calc_average_prediction(perm: torch.Tensor, outputs: torch.Tensor, mesh: data.Mesh) -> torch.Tensor:
        """
        Calculate every prediction for every vertex in mesh over multiple walks.

        :param perm: tensor containing the indices of the vertices for the corresponding output
        :param outputs: tensor containing the prediction for vertices the vertices
        :param mesh: mesh object
        :return: average prediction per vertex
        """
        pred = torch.zeros((mesh.vertices.shape[0], outputs.shape[2]), device=outputs.device)
        pred_count = 1e-10 * torch.ones((mesh.vertices.shape[0],), device=outputs.device)
        for walk in range(outputs.shape[0]):
            pred[perm[walk]] += outputs[walk]
            pred_count[perm[walk]] += 1

        return (pred.T / pred_count).T

    @staticmethod
    def segmentation_accuracy(mesh: data.Mesh, av_pred: torch.Tensor) -> int:
        """
        Calculate the segmentation accuracy of an average prediction vector per vertex. A vertex is considered correctly
        classified if the corresponding soft vertex label is not zero for the predicted class.

        :param mesh: mesh object
        :param av_pred: average prediction per vertex
        :return: number of correctly classified vertices
        """
        best_pred = torch.argmax(av_pred, dim=-1)
        correct = 0
        for i in range(best_pred.shape[0]):
            if mesh.soft_labels[i, best_pred[i]] != 0:
                correct += 1

        return correct

    @staticmethod
    def classification_accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Calculates the mean accuracy of the classification of the meshes over one or more walks.

        :param outputs: prediction vector
        :param labels: labels
        :return: number of correctly classified meshes
        """
        _, predicted = torch.max(outputs.mean(dim=1).data, -1)
        correct = (predicted == labels).sum()
        return correct.item()

    def test(self, model: nn.Module = None, dataset: data.DataLoader = None, finetune: bool = False) -> float:
        """
        Executes the test function depending on the dataset (classification or segmentation)

        :param model: examined model. Defaults to the base model
        :param dataset: used dataset. Defaults to test dataset
        :param finetune: if True, finetune on the given dataset. Defaults to False
        :return:number of correctly classified meshes or vertices
        """
        if model is None:
            model = self.model

        if dataset is None:
            dataset = self.test_set

        model = model.encoder if self.pretrain else model
        if self.opt.dataset_mode == data.SegmentationData:
            return self.test_segmentation(model, dataset, finetune)
        else:
            return self.test_classification(model, dataset, finetune)

    def online_finetune(self, seq: torch.Tensor, grad: bool = False) -> torch.Tensor:
        """
        Use the online finetuner

        :param seq: input to the online finetuner
        :param grad: if True, update the weights in online finetuner. Defaults to False.
        :return: output of the online finetuner
        """
        if grad:
            outputs = self.finetuner(seq)
        else:
            with torch.no_grad():
                outputs = self.finetuner(seq)
        return outputs

    def visualize(self) -> None:
        """
        Visualize the attention weights of the model on the vertices of the meshes in the test dataset. It saves the
        visualized mesh as .ply file and the normalized mesh including the faces as .obj file. Both are saved in folder

        output/visualization/dataset/model/train_approach/name/
        """

        norm = options.normalizer_options[self.opt.normalizer]
        c_map = util.get_color_map(self.opt.color_map)
        # don't adjust gradients
        with torch.no_grad():
            with tqdm(total=min(self.opt.amount, len(self.test_set))):
                for batch_id, seq in enumerate(self.test_set):
                    inputs, labels, pads, perm = self.get_sequence_data(seq)
                    outputs, attn_weights_mat = self.model(inputs, pads)
                    batch_size, walks = attn_weights_mat.shape[:2]
                    for i in range(batch_size):
                        mesh = seq['mesh'][i]
                        # compute attention colors
                        att_colors = util.compute_attention_colors(attn_weights_mat[i], perm[i], mesh.vertices.shape[0],
                                                                   c_map, normalizer=norm)
                        save_path = f"{os.path.join(util.get_vis_path(self.opt), mesh.filename)}"
                        # write ply file with vertex colors
                        util.write_mesh_colors(mesh, save_path, att_colors)
                        # write obj file with faces
                        util.save_obj(mesh, save_path)

    def attack(self) -> List[str]:
        """
        Reimplementation of the adversarial attack on the meshes by Belder et al. (https://arxiv.org/pdf/2202.07453.pdf)
        It uses the gradients of the imitation model in respect to the input vertices.
        If the imitation model correctly classifies a mesh, the vertices are moved in the opposite direction of the
        gradients and multiplied by a constant factor opt.gradient_weight. This is repeated until the mesh is
        misclassified or the maximum amount the iterations opt.max_attack_iterations is reached.

        :return: a list with the paths of the modified meshes.
        """
        modified_meshes = []
        criterion = self.get_criterion()
        rng = tqdm(range(len(self.test_set.dataset)))
        iterations = []
        l2_loss = []
        state_dict = copy.deepcopy(self.imi_model.state_dict())
        for j in rng:
            original_mesh = copy.deepcopy(self.test_set.dataset[j]['mesh'])
            iter_mean = np.mean(iterations) if iterations else 0
            l2_mean = np.mean(l2_loss) if l2_loss else 0
            for iteration in range(self.opt.max_attack_iterations):
                # messy hack to prevent model from training but still use its gradient: always restore old checkpoints
                self.imi_model.load_state_dict(state_dict)
                # get new random walk
                seq = self.test_set.dataset[j]
                mesh = seq['mesh']
                inputs, label, pads, perm = self.get_sequence_data(seq)
                # recalculate the mesh signature
                if self.opt.eig_basis > 0:
                    mesh.signature = mesh.calc_signature()
                self.optimizer.zero_grad()
                # activate grads for input
                inputs.requires_grad = True
                prediction = self.imi_model(inputs, pads)
                one_hot_label = self.get_one_hot_label(label)
                pred_softmax = F.softmax(prediction[0], dim=0)
                diff = torch.mean(torch.abs(pred_softmax - one_hot_label), dim=0).item()
                # if wrong classified
                if torch.argmax(pred_softmax, dim=0) != label:
                    # mesh has been modified
                    if iteration != 0:
                        iterations.append(iteration)
                        modified_meshes.append(mesh.filename)
                        l2 = F.mse_loss(torch.from_numpy(mesh.vertices), torch.from_numpy(original_mesh.vertices))
                        l2_loss.append(l2.item())
                    break

                # calc loss
                pred_logsoftmax = F.log_softmax(prediction, dim=1)
                loss = criterion(pred_logsoftmax[0], one_hot_label)
                self.update_optimizer(loss, self.optimizer)

                # calc gradient of vertices
                gradient = inputs.grad[0] * self.opt.gradient_weight
                mesh.vertices[perm.detach().cpu().numpy()] += gradient.detach().cpu().numpy()[:, :3]

                rng.set_postfix_str(
                    f'misclassified: {len(iterations)}, mean iter: {iter_mean}, L2: {l2_mean}, diff: {np.round(diff, 5)}')
        return modified_meshes

    def get_one_hot_label(self, label: int) -> torch.Tensor:
        """
        Calculates a one hot encoded label, which is a tensor containing only zeros except for a one in the label index.

        :param label: label index
        :return: one hot encoded label
        """
        one_hot_label = torch.zeros((self.opt.n_classes,), dtype=torch.float, device=self.opt.device)
        one_hot_label[label] = 1.
        return one_hot_label
