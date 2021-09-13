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
import sys
import numpy as np


# learning_type = simple, joint, exemplar, lwf
# forgetting

# 20, 25, 30, 35, 40
# 20, 5,  5,  5,  5
def rand_choice(choices, size, count):
    r = np.arange(size)
    r = np.delete(r, choices)
    temp = np.random.choice(len(r), size=count, replace=False)
    return r[temp]


# noinspection PyUnboundLocalVariable
def training(opt, n_class=None, flag=False, classes=None, except_samples = []):

    _fe = opt.learning_type == 'forgetting' or opt.learning_type == 'exemplar'
    assert (opt.exemplar_num is not None and opt.exemplar_num > 0)

    if opt.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            npoints=opt.num_points, num_class=n_class)

        test_dataset = ShapeNetDataset(
            root=opt.dataset,
            classification=True,
            split='test',
            npoints=opt.num_points,
            data_augmentation=False, num_class=n_class)
    elif opt.dataset_type == 'modelnet40':
        if not opt.is_h5:
            dataset = ModelNetDataset(
                root=opt.dataset,
                npoints=opt.num_points,
                split='train_files', num_class=n_class)

            test_dataset = ModelNetDataset(
                root=opt.dataset,
                split='test_files',
                npoints=opt.num_points,
                data_augmentation=False, num_class=n_class)

        else:
            dataset = ModelNet40(root=opt.dataset, partition='train', num_points=opt.num_points)

            test_dataset = ModelNet40(root=opt.dataset, partition='test', num_points=opt.num_points)

    else:
        exit('wrong dataset type')

    num_classes = len(dataset.classes)

    if classes is not None:
        if flag:
            temp = rand_choice(classes, 40, n_class)
            classes = np.concatenate((classes, temp))
            dataset.filter(classes=classes)
            test_dataset.filter(classes=classes)
        else:
            temp = rand_choice(classes, 40, opt.step_num_class)
            classes = np.concatenate((classes, temp))
            if _fe:
                dataset.filter(temp, except_samples)
                test_dataset.filter(classes)
            else:
                dataset.filter(classes)
                test_dataset.filter(classes)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.workers))

    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    classifier = PointNetCls(k=num_classes, feature_transform=opt.feature_transform)

    epochs = opt.nepoch
    if _fe and not flag:
        print('loading previous model')
        epochs = 30
        classifier.load_state_dict(torch.load('%s/cls_model_%s.pth' % (opt.outf, opt.learning_type)))

    if opt.model != '':
        classifier.load_state_dict(torch.load(opt.model))

    optimizer = optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    classifier.cuda()

    if flag:
        print(str(classifier))

    print(len(dataset), len(test_dataset))
    print('classes', len(dataset.classes))

    num_batch = len(dataset) / opt.batchSize

    blue = lambda x: '\033[94m' + x + '\033[0m'

    test_acc = np.zeros(epochs)
    test_loss = np.zeros(epochs)
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
                if opt.feature_transform:
                    loss += feature_transform_regularizer(trans_feat) * 0.001
                loss.backward()
                optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()

            pbar.update(1)
            pbar.postfix = f'loss: {round(loss.item(), 2)}, acc: {round(correct.item() / float(opt.batchSize), 2)}'

        pbar.close()

        if opt.test_epoch > 0 and (epoch + 1) % opt.test_epoch == 0 or (epoch + 1) == epochs:
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

            test_acc[epoch] = total_correct / float(total_testset)
            test_loss[epoch] = total_loss / float(total_testset)
            print('[Epoch %d] %s loss: %f accuracy: %f\n' % (
                epoch + 1, blue('test'), test_loss[epoch], test_acc[epoch]))

        torch.save(classifier.state_dict(), '%s/cls_model_%d.pth' % (opt.outf, epoch))

    if opt.test_epoch > 0:
        np.save(f'results/accuracy_cls{n_class}_{opt.learning_type}.npy', test_acc)
        np.save(f'results/loss_cls{n_class}_{opt.learning_type}.npy', test_loss)

    if _fe:
        torch.save(classifier.state_dict(), '%s/cls_model_%s.pth' % (opt.outf, opt.learning_type))

    return num_classes, classes, test_acc[-1]


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
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--is_h5', type=bool, default=False,
                        help='is h5')
    parser.add_argument('--test_epoch', type=int, default=1,
                        help='1 for all epochs, 0 for last epoch, n for each n epoch')
    parser.add_argument('--exemplar_num', type=int, default=1, help='iif learning_type is exemplar')
    parser.add_argument('--progress', type=bool, default=False,
                        help='has new progress?')

    opt = parser.parse_args()
    print(opt)

    # opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Manual Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)

    flag = True

    if opt.learning_type == "simple":
        training(opt=opt, flag=flag)
    else:
        n_class = opt.start_num_class
        classes = np.array([], dtype=int)
        accuracies = []

        while True:

            num_classes, classes, accuracy = training(opt, n_class, flag, classes)
            accuracies.append(accuracy)

            flag = False
            n_class += opt.step_num_class
            if n_class > num_classes:
                np.save(f'{opt.learning_type}.npy', accuracies)
                break
