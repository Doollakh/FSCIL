from __future__ import print_function
import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

from pointnet.dataset import ShapeNetDataset, ModelNetDataset, ModelNet40
from pointnet.losses import KnowlegeDistilation, PointNetLoss
from pointnet.model import PointNetCls, PointNetLwf
# learning_type = simple, joint, exemplar, lwf, bExemplar
# forgetting
from utils.general import increment_path
from utils.plotcm import plot_confusion_matrix


class Learning:

    def __init__(self, options, s_d):
        self.opt = options
        self.save_dir = s_d
        self.n_class = 0
        self.classes = None
        self.except_samples = []
        self.accuracies = None
        self.order = None
        self.num_classes = 0

    def start(self):

        _fe = self.opt.learning_type == 'forgetting' \
              or self.opt.learning_type == 'exemplar' \
              or self.opt.learning_type == 'bExemplar' \
              or self.opt.learning_type == 'bCandidate'

        lwf = False if not _fe else self.opt.lwf
        assert (self.opt.exemplar_num is not None and self.opt.exemplar_num > 0)
        flag = True
        if self.opt.continue_from is True:
            assert self.opt.learning_type != 'simple'

        self.order = [2, 3, 4, 10, 14, 17, 19, 21, 22, 26, 27, 28, 29, 30, 31, 32, 33, 35, 36, 39, 5, 16, 23, 25, 37, 9,
                      12, 13, 20, 24, 0, 1, 6, 34, 38, 7, 8, 11, 15, 18]
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

                self.train(flag, _fe, skip, lwf)

                flag = False
                self.n_class += self.opt.step_num_class
                if self.n_class > self.num_classes:
                    num_exemplar = '_' + str(
                        self.opt.exemplar_num) if _fe and not self.opt.learning_type == 'forgetting' else ''
                    np.save(f'{self.opt.learning_type}{num_exemplar}.npy', self.accuracies)
                    break

    # 20, 25, 30, 35, 40
    # 20, 5,  5,  5,  5
    def rand_choice(self, size, count):
        r = np.arange(size)
        r = np.delete(r, self.classes)
        temp = np.random.choice(len(r), size=count, replace=False)
        return r[temp]

    def select_dataset(self):
        if self.opt.dataset_type == 'modelnet40':
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
                few = self.opt.few_shots if self.opt.f else None
                dataset = ModelNet40(root=self.opt.dataset, partition='train', num_points=self.opt.num_points, few=few
                                     , from_candidates=self.opt.learning_type == 'bCandidate')

                test_dataset = ModelNet40(root=self.opt.dataset, partition='test', num_points=self.opt.num_points,
                                          few=None)

            self.num_classes = len(dataset.classes)
            dataset.set_order(self.order)
            test_dataset.set_order(self.order)

        else:
            exit('wrong dataset type')

        return dataset, test_dataset

    def find_best_samples(self, n):
        arr = np.load('utils/best_exemplar.npy')[:, :n]
        return arr[self.classes].reshape(len(self.classes) * n)

    def find_first_occurrence(self, dataset, n):
        visited = np.zeros(self.num_classes, dtype=int)
        visited_in = np.zeros((self.num_classes, n), dtype=int)
        for i, item in enumerate(dataset.label):
            if visited[item] != n:
                visited_in[item, visited[item]] = i
                visited[item] += 1
        return visited_in[self.classes].reshape(len(self.classes) * n)

    def train(self, flag=False, _fe=False, skip=False, lwf=False):

        dataset, test_dataset = self.select_dataset()

        if self.classes is not None:
            if flag:
                # temp = self.rand_choice(self.num_classes, self.n_class)
                self.classes = np.arange(self.n_class)
                dataset.filter(self.classes)
                test_dataset.filter(self.classes)
            else:
                if self.opt.learning_type == 'exemplar':
                    self.except_samples = self.find_first_occurrence(dataset, self.opt.exemplar_num)
                if self.opt.learning_type == 'bExemplar':
                    self.except_samples = self.find_best_samples(self.opt.exemplar_num)
                    print('old_samples: ', self.except_samples)

                self.classes = np.arange(self.n_class)
                temp = self.classes[-self.opt.step_num_class:]
                if _fe:
                    cand_ids = None
                    if self.opt.learning_type == 'bCandidate':
                        cand_ids = np.arange(1, 105, step=3)[:len(self.classes)-self.opt.step_num_class]
                        print('cand_ids: ', cand_ids)

                    dataset.filter(temp, self.except_samples, cand_ids)
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

        if lwf and not flag:
            classifier = PointNetCls(k=self.n_class - self.opt.step_num_class,
                                     feature_transform=self.opt.feature_transform)
            # todo save classifier after first task and save new_model and second task
        else:
            classifier = PointNetCls(k=self.n_class, feature_transform=self.opt.feature_transform)

        epochs = opt.nepoch
        if _fe and not flag:
            epochs = 30

        if skip:
            print('loading previous model')
            classifier.feat.load_state_dict(
                torch.load('%s/cls_model_%s_%d.pth' % (self.opt.dir_pretrained, self.opt.learning_type, self.n_class)))
        elif _fe and not flag:
            print('loading previous model')
            classifier.feat.load_state_dict(
                torch.load('%s/cls_model_%s_%d.pth' % (
                    self.save_dir, self.opt.learning_type, self.n_class - self.opt.step_num_class)))

        if self.opt.model != '':
            classifier.load_state_dict(torch.load(self.opt.model))

        optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        classifier.cuda()

        if _fe:
            self.compute_best_samples(classifier.feat)

        if flag:
            print(str(classifier))

        print(len(dataset), len(test_dataset))
        print('classes', len(dataset.classes))

        blue = lambda x: '\033[94m' + x + '\033[0m'

        test_acc = np.zeros(epochs)
        test_loss = np.zeros(epochs)
        if skip:
            print('skipping this stage ...')
            test_loss[-1], test_acc[-1] = self.test(classifier, testdataloader, False, self.n_class)
            self.accuracies.append(test_acc[-1])
            print('%s loss: %f accuracy: %f\n' % (blue('test'), test_loss[-1], test_acc[-1]))
            print()
            torch.save(classifier.feat.state_dict(),
                       '%s/cls_model_%s_%d.pth' % (self.save_dir, self.opt.learning_type, self.n_class))
            return

        point_loss = PointNetLoss().cuda()
        if lwf and not flag:
            kd_loss = KnowlegeDistilation(T=float(self.opt.dist_temperature)).cuda()
            shared_model = classifier.copy()
            new_model = PointNetLwf(shared_model, old_k=self.n_class - self.opt.step_num_class,
                                    new_k=self.opt.step_num_class).cuda()
            print(new_model)

        for epoch in range(epochs):
            scheduler.step()
            n = len(dataloader)
            pbar = tqdm(total=n, desc=f'Epoch: {epoch + 1}/{epochs}  ', ncols=110)
            for i, data in enumerate(dataloader, 0):
                points, target = data
                if len(points) != 1:
                    # target = target[:, 0]
                    points.transpose_(2, 1)
                    points, target = points.cuda(), target.cuda()
                    optimizer.zero_grad()
                    if lwf and not flag:
                        classifier.eval()
                    else:
                        classifier.train()
                    pred, _, trans_feat = classifier(points)
                    if lwf and not flag:
                        new_model.train()
                        old_pred, new_pred, _, trans_feat = new_model(points)
                        loss = point_loss(new_pred, target, trans_feat, self.opt.feature_transform)
                        kdl = kd_loss(pred, old_pred)
                        loss += kdl * self.opt.dist_factor
                        classifier_ = new_model
                    else:
                        classifier_ = classifier
                        loss = point_loss(pred, target, trans_feat, self.opt.feature_transform)

                    loss.backward()
                    optimizer.step()
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()

                pbar.update(1)
                acc = round(correct.item() / float(self.opt.batchSize), 2)
                if lwf and not flag:
                    pbar.postfix = f'loss: {round(loss.item(), 2)}, acc: {acc}, kd_loss: {round(kdl.item(), 2)}'
                else:
                    pbar.postfix = f'loss: {round(loss.item(), 2)}, acc: {acc}'

            pbar.close()

            if self.opt.test_epoch > 0 and (epoch + 1) % self.opt.test_epoch == 0 or (epoch + 1) == epochs:
                test_loss[epoch], test_acc[epoch] = self.test(classifier_, testdataloader, lwf and not flag,
                                                              self.n_class)
                print('[Epoch %d] %s loss: %f accuracy: %f\n' % (
                    epoch + 1, blue('test'), test_loss[epoch], test_acc[epoch]))

            if self.opt.save_after_epoch:
                torch.save(classifier_.feat.state_dict(),
                           '%s/cls_model_%d_%d.pth' % (self.save_dir, epoch, self.n_class))

        if self.opt.test_epoch > 0:
            np.save(f'{self.save_dir}/results/accuracy_cls{self.n_class}_{self.opt.learning_type}.npy', test_acc)
            np.save(f'{self.save_dir}/results/loss_cls{self.n_class}_{self.opt.learning_type}.npy', test_loss)

        torch.save(classifier_.feat.state_dict(),
                   '%s/cls_model_%s_%d.pth' % (self.save_dir, self.opt.learning_type, self.n_class))

        if _fe:
            self.accuracies.append(test_acc[-1])

    @staticmethod
    def test(classifier, testdataloader, lwf, n_class):
        cmt = torch.zeros(n_class, n_class, dtype=torch.int64)
        total_loss = 0
        total_correct = 0
        total_testset = 0
        for i, data in tqdm(enumerate(testdataloader, 0)):
            points, target = data
            # target = target[:, 0]
            points.transpose_(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            if not lwf:
                pred, _, _ = classifier(points)
            else:
                _, pred, _, _ = classifier(points)

            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            stacked = torch.stack((target.data, pred_choice), dim=1).cpu()
            for p in stacked:
                tl, pl = p.tolist()
                cmt[tl, pl] = cmt[tl, pl] + 1
            loss = F.nll_loss(pred, target)
            total_loss += loss.item()
            total_correct += correct.item()
            total_testset += points.size()[0]
        plt.figure(figsize=(10, 10))
        # plot_confusion_matrix(cmt, np.arange(n_class))

        return total_loss / float(total_testset), total_correct / float(total_testset)

    def compute_best_samples(self, classifier):
        if self.opt.learning_type == 'bExemplar':
            best_exemplar = Path('utils/best_exemplar.npy')
            if not best_exemplar.exists():
                dataset, _ = self.select_dataset()
                loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=self.opt.batchSize,
                    shuffle=False,
                    num_workers=int(self.opt.workers))
                self._compute_best_samples(classifier, loader, best_exemplar)

    @staticmethod
    def _compute_best_samples(classifier, dataloader, best_exemplar):
        print('computing best samples...')
        for i, data in tqdm(enumerate(dataloader, 0)):
            points, target = data
            points = points.transpose(2, 1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _, _ = classifier(points)
            X = np.concatenate((X, pred.cpu().detach().numpy()), 0) if i != 0 else pred.cpu().detach().numpy()
            y = np.concatenate((y, target.cpu().detach().numpy()), 0) if i != 0 else target.cpu().detach().numpy()

        cos = lambda in1, in2: np.dot(in1, in2) / (np.linalg.norm(in1) * np.linalg.norm(in2))

        uniq = np.unique(y, return_counts=True)[1]
        max_s = np.max(uniq)
        results = np.zeros((40, max_s), dtype=int)
        # avg = np.zeros((40, 1024))
        for i in range(40):
            w = np.where(y == i)[0]
            cX = X[w]
            n = len(cX)
            r = np.zeros(n)
            avg = np.average(cX, axis=0)
            for j, item in enumerate(cX):
                r[j] = cos(avg, item)
            args = np.argsort(r)[::-1][:n]
            results[i, 0:n] = w[args]

        np.save(best_exemplar, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=1024, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=250, help='number of epochs to train for')
    parser.add_argument(
        '--learning_type', type=str, default='simple', help='')
    parser.add_argument('--lwf', action='store_true', help='is lwf')
    parser.add_argument(
        '--start_num_class', type=int, help='', default=20)
    parser.add_argument(
        '--step_num_class', type=int, help='', default=5)
    parser.add_argument(
        '--manualSeed', type=int, help='', default=1)
    parser.add_argument(
        '--dist_temperature', type=int, default=1, help='distillation temperature')
    parser.add_argument(
        '--dist_factor', type=float, default=0.4, help='distillation factor')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--f', action='store_true', help='')
    parser.add_argument(
        '--few_shots', type=int, default=5, help='')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--is_h5', type=bool, default=True,
                        help='is h5')
    parser.add_argument('--save_after_epoch', action='store_true',
                        help='save model after each epoch')
    parser.add_argument('--test_epoch', type=int, default=1,
                        help='1 for all epochs, 0 for last epoch, n for each n epoch')
    parser.add_argument('--exemplar_num', type=int, default=1, help='iif learning_type is exemplar')
    parser.add_argument('--continue_from', type=int, default=None, help='')
    parser.add_argument('--dir_pretrained', type=str, default='cls', help='load pretrained model')
    parser.add_argument('--progress', type=bool, default=True,
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
