from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset, ModelNet40
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from pathlib import Path

# learning_type = simple, joint, exemplar, lwf
# forgetting
from utils.general import increment_path


class Learning:

    def __init__(self, options, s_d):
        self.opt = options
        self.save_dir = s_d
        self.n_class = 0
        self.classes = None
        self.except_samples = []
        self.accuracies = None
        self.num_classes = 0

    def start(self):

        _fe = self.opt.learning_type == 'forgetting' or self.opt.learning_type == 'exemplar'
        assert (self.opt.exemplar_num is not None and self.opt.exemplar_num > 0)
        flag = True
        if self.opt.continue_from is True:
            assert self.opt.learning_type != 'simple'

        if self.opt.learning_type == "simple":
            self.classes = None
            self.train(flag)
        else:
            self.n_class = opt.start_num_class
            self.classes = np.array([], dtype=int)
            self.accuracies = []

            while True:

                skip = False
                if self.opt.continue_from is not None:
                    skip = self.opt.continue_from > self.n_class

                self.train(flag, _fe, skip)

                flag = False
                self.n_class += self.opt.step_num_class
                if self.n_class > self.num_classes:
                    np.save(f'{self.opt.learning_type}.npy', self.accuracies)
                    break

    # 20, 25, 30, 35, 40
    # 20, 5,  5,  5,  5
    def rand_choice(self, size, count):
        r = np.arange(size)
        r = np.delete(r, self.classes)
        temp = np.random.choice(len(r), size=count, replace=False)
        return r[temp]

    def select_dataset(self):
        if self.opt.dataset_type == 'shapenet':
            dataset = ShapeNetDataset(
                root=self.opt.dataset,
                classification=True,
                npoints=self.opt.num_points)

            test_dataset = ShapeNetDataset(
                root=self.opt.dataset,
                classification=True,
                split='test',
                npoints=self.opt.num_points,
                data_augmentation=False)
        elif self.opt.dataset_type == 'modelnet40':
            if not self.opt.is_h5:
                dataset = ModelNetDataset(
                    root=self.opt.dataset,
                    npoints=self.opt.num_points,
                    split='train_files')

                test_dataset = ModelNetDataset(
                    root=self.opt.dataset,
                    split='test_files',
                    npoints=self.opt.num_points,
                    data_augmentation=False)

            else:
                dataset = ModelNet40(root=self.opt.dataset, partition='train', num_points=self.opt.num_points)

                test_dataset = ModelNet40(root=self.opt.dataset, partition='test', num_points=self.opt.num_points)

            self.num_classes = len(dataset.classes)

        else:
            exit('wrong dataset type')

        return dataset, test_dataset

    def find_first_occurrence(self, dataset, n):
        visited = np.zeros(self.num_classes, dtype=int)
        visited_in = np.zeros((self.num_classes, n), dtype=int)
        for i, item in enumerate(dataset.label):
            if visited[item] != n:
                visited_in[item, visited[item]] = i
                visited[item] += 1
        return visited_in[self.classes].reshape(len(self.classes) * n)

    def train(self, flag=False, _fe=False, skip=False):

        dataset, test_dataset = self.select_dataset()

        if self.classes is not None:
            if flag:
                temp = self.rand_choice(self.num_classes, self.n_class)
                self.classes = np.concatenate((self.classes, temp))
                dataset.filter(self.classes)
                test_dataset.filter(self.classes)
            else:
                if self.opt.learning_type == 'exemplar':
                    self.except_samples = self.find_first_occurrence(dataset, self.opt.exemplar_num)
                    # print('old_samples: ', self.except_samples)
                temp = self.rand_choice(self.num_classes, self.opt.step_num_class)
                self.classes = np.concatenate((self.classes, temp))
                if _fe:
                    dataset.filter(temp, self.except_samples)
                    test_dataset.filter(self.classes)
                else:
                    dataset.filter(self.classes)
                    test_dataset.filter(self.classes)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=False,
            num_workers=int(self.opt.workers))

        testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.opt.batchSize,
            shuffle=False,
            num_workers=int(self.opt.workers))

        try:
            (self.save_dir / 'results' if self.opt.test_epoch > 0 else self.save_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        classifier = PointNetCls(k=self.num_classes, feature_transform=self.opt.feature_transform)

        epochs = opt.nepoch
        if _fe and not flag:
            epochs = 30

        if skip and _fe:
            print('loading previous model')
            classifier.load_state_dict(
                torch.load('%s/cls_model_%s_%d.pth' % (self.opt.dir_pretrained, self.opt.learning_type, self.n_class)))
        elif _fe and not flag:
            print('loading previous model')
            classifier.load_state_dict(
                torch.load('%s/cls_model_%s_%d.pth' % (
                self.save_dir, self.opt.learning_type, self.n_class - self.opt.step_num_class)))

        if self.opt.model != '':
            classifier.load_state_dict(torch.load(self.opt.model))

        optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        classifier.cuda()

        if flag:
            print(str(classifier))

        print(len(dataset), len(test_dataset))
        print('classes', len(dataset.classes))

        blue = lambda x: '\033[94m' + x + '\033[0m'

        test_acc = np.zeros(epochs)
        test_loss = np.zeros(epochs)
        if skip:
            print('skipping this stage ...')
            test_loss[-1], test_acc[-1] = self.test(classifier, testdataloader)
            self.accuracies.append(test_acc[-1])
            print('%s loss: %f accuracy: %f\n' % (blue('test'), test_loss[-1], test_acc[-1]))
            print()
            return

        for epoch in range(epochs):
            scheduler.step()
            n = len(dataloader)
            pbar = tqdm(total=n, desc=f'Epoch: {epoch + 1}/{epochs}  ', ncols=110)
            for i, data in enumerate(dataloader, 0):
                points, target = data
                if len(points) != 1:
                    target = target[:, 0]
                    points = points.transpose(2, 1)
                    points, target = points.cuda(), target.cuda()
                    optimizer.zero_grad()
                    classifier = classifier.train()
                    pred, trans, trans_feat = classifier(points)
                    loss = F.nll_loss(pred, target)
                    if self.opt.feature_transform:
                        loss += feature_transform_regularizer(trans_feat) * 0.001
                    loss.backward()
                    optimizer.step()
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()

                pbar.update(1)
                acc = round(correct.item() / float(self.opt.batchSize), 2)
                pbar.postfix = f'loss: {round(loss.item(), 2)}, acc: {acc}'

            pbar.close()

            if self.opt.test_epoch > 0 and (epoch + 1) % self.opt.test_epoch == 0 or (epoch + 1) == epochs:
                test_loss[epoch], test_acc[epoch] = self.test(classifier, testdataloader)
                print('[Epoch %d] %s loss: %f accuracy: %f\n' % (
                    epoch + 1, blue('test'), test_loss[epoch], test_acc[epoch]))

            if self.opt.save_after_epoch:
                torch.save(classifier.state_dict(), '%s/cls_model_%d_%d.pth' % (self.save_dir, epoch, self.n_class))

        if self.opt.test_epoch > 0:
            np.save(f'{self.save_dir}/results/accuracy_cls{self.n_class}_{self.opt.learning_type}.npy', test_acc)
            np.save(f'{self.save_dir}/results/loss_cls{self.n_class}_{self.opt.learning_type}.npy', test_loss)

        torch.save(classifier.state_dict(),
                   '%s/cls_model_%s_%d.pth' % (self.save_dir, self.opt.learning_type, self.n_class))
        if _fe:
            self.accuracies.append(test_acc[-1])

    @staticmethod
    def test(classifier, testdataloader):
        total_loss = 0
        total_correct = 0
        total_testset = 0
        for i, data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            target = target[:, 0]
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            loss = F.nll_loss(pred, target)
            total_loss += loss.item()
            total_correct += correct.item()
            total_testset += points.size()[0]

        return total_loss / float(total_testset), total_correct / float(total_testset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument(
        '--learning_type', type=str, default='simple', help='')
    parser.add_argument(
        '--start_num_class', type=int, help='', default=20)
    parser.add_argument(
        '--step_num_class', type=int, help='', default=5)
    parser.add_argument(
        '--manualSeed', type=int, help='', default=1)
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--is_h5', type=bool, default=False,
                        help='is h5')
    parser.add_argument('--save_after_epoch', type=bool, default=False,
                        help='save model after each epoch')
    parser.add_argument('--test_epoch', type=int, default=1,
                        help='1 for all epochs, 0 for last epoch, n for each n epoch')
    parser.add_argument('--exemplar_num', type=int, default=1, help='iif learning_type is exemplar')
    parser.add_argument('--continue_from', type=int, default=None, help='')
    parser.add_argument('--dir_pretrained', type=str, default='cls', help='load pretrained model')
    parser.add_argument('--progress', type=bool, default=False,
                        help='has new progress?')

    opt = parser.parse_args()
    print(opt)

    save_dir = increment_path(Path(opt.outf) / opt.name, exist_ok=opt.exist_ok)

    # opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Manual Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)

    learning = Learning(opt, save_dir)
    learning.start()
