from __future__ import print_function
import argparse
import copy
import random
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from focal_loss.focal_loss import FocalLoss
import matplotlib.pyplot as plt 
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from tqdm import tqdm

import shutil


from pointnet.dataset import ModelNetDataset, ModelNet40, ScanObjects, ModelNet40_ScanObjects, ShapeNet
from pointnet.losses import KnowlegeDistilation, PointNetLoss
from pointnet.model import PointNetCls, PointNetLwf
from pointnet.bests import simple_clustring, spectral_clustring, spectral_clustring_mod
# learning_type = simple, joint, exemplar, lwf, bExemplar
# forgetting
from utils.general import increment_path
from utils.loss_functions import AngularPenaltySMLoss
from utils.log import Log


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
        self.diff = 0

    def start(self):
        ############
        if self.opt.best_type == 'disjoint' :
          self.opt.n_cands += 5
        #############

        _fe = self.opt.learning_type == 'forgetting' \
              or self.opt.learning_type == 'exemplar' \
              or self.opt.learning_type == 'bExemplar' \
              or self.opt.learning_type == 'bCandidate'

        lwf = False if not _fe else self.opt.lwf
        assert (self.opt.exemplar_num is not None and self.opt.exemplar_num > 0)
        flag = True
        if self.opt.continue_from is True:
            assert self.opt.learning_type != 'simple'
        
        # select ordering type
        if self.opt.dataset_type == 'modelnet40':      
            if self.opt.order == 'fscil_order':
                self.order  = [8,30,0,4,2,37,22,33,35,5,21,36,26,25,7,12,14,23,16,17,28,3,9,34,15,20,18,11,1,29,19,31,13,27,39,32,24,38,10,6]
                self.class_names = ['chair','sofa','airplane','bookshelf','bed','vase','monitor','table','toilet','bottle',
                            'mantel','tv stand','plant','piano','car','desk','dresser','night stand','glass box',
                            'guitar','range hood','bench','cone','tent','flower pot','laptop','keyboard','curtain',
                            'bathtub','sink','lamp','stairs','door','radio','xbox','stool','person','wardrobe','cup','bowl']
            
            elif self.opt.order == 'orginal_order':
                self.order = [i for i in range(40)]
                self.class_names = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower pot','glass box','guitar','keyboard','lamp','laptop','mantel','monitor','night stand','person','piano','plant','radio','range hood','sink','sofa','stairs','stool','table','tent','toilet','tv stand','vase','wardrobe','xbox']
            
            else:
                print("You entered wrong ordering type!")
                exit()
            
            np.save('./misc/class_names.npy', np.array(self.class_names))

        elif self.opt.dataset_type == 'modelnet40_scanobjects':
            self.order = [i for i in range(36)]
        elif self.opt.dataset_type == 'shapenet':
            self.order = [i for i in range(55)]
            self.class_names = ['airplane', 'bag', 'basket' , 'bathtub' , 'bed' , 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellphone', 'chair', 'clock', 'dishwasher', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'keyboard', 'knife', 'lamp', 'laptop', 'mailbox', 'microphone', 'microwave', 'monitor', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket', 'skateboard', 'sofa', 'speaker', 'stove', 'table', 'telephone', 'tin_can', 'tower', 'train', 'vessel', 'washer']

        log_class.data['config']['order'] = self.opt.order


        if self.opt.learning_type == "simple":
            self.classes = None
            self.train(flag)
        else:
            self.n_class = opt.start_num_class
            self.classes = np.array([], dtype=int)

            self.accuracies = []
            
            stage_id = 0
            temp_flag = True
            while True:

                skip = False
                if self.opt.continue_from is not None:
                    skip = self.opt.continue_from > self.n_class

                if flag == False:
                    stage_id += 1

                self.train(flag, _fe, skip, lwf, stage_id=stage_id)

                flag = False
                self.n_class += self.opt.step_num_class
                if self.num_classes+self.opt.step_num_class == self.n_class:
                    num_exemplar = '_' + str(
                        self.opt.exemplar_num) if _fe and not self.opt.learning_type == 'forgetting' else ''
                    np.save(f'{self.opt.learning_type}{num_exemplar}.npy', self.accuracies)
                    break

                if self.n_class > self.num_classes:
                    for i in range(self.n_class-self.num_classes+1):
                        if self.n_class-i <= self.num_classes:
                            self.diff += i-1
                            self.n_class -= i-1
                            temp_flag = not temp_flag
                            break
                    if temp_flag:
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

    def select_dataset(self, stage_id):
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
                if stage_id == 0:
                  dataset = ModelNet40(root=self.opt.dataset, partition='train', num_points=self.opt.num_points, few=few
                                      , from_candidates=False,n_cands=self.opt.n_cands,cands_path=self.opt.cands_path, aligned=self.opt.aligned)

                  test_dataset = ModelNet40(root=self.opt.dataset, partition='test', num_points=self.opt.num_points,
                                            few=None, aligned=self.opt.aligned)

                else:

                  dataset = ModelNet40(root=self.opt.dataset, partition='train', num_points=self.opt.num_points, few=few
                                      , from_candidates=self.opt.learning_type == 'bCandidate',n_cands=self.opt.n_cands,cands_path=self.opt.cands_path, aligned=self.opt.aligned)

                  test_dataset = ModelNet40(root=self.opt.dataset, partition='test', num_points=self.opt.num_points,
                                            few=None, aligned=self.opt.aligned)

            self.num_classes = len(dataset.classes)
            dataset.set_order(self.order)
            test_dataset.set_order(self.order)

        elif self.opt.dataset_type == 'scanobjects':
            print('(Info) Reading Scan object')
            few = self.opt.few_shots if self.opt.f else None
            dataset = ScanObjects(root=self.opt.dataset, partition='train', num_points=self.opt.num_points, few=few
                                     , from_candidates=self.opt.learning_type == 'bCandidate',n_cands=self.opt.n_cands,cands_path=self.opt.cands_path)

            test_dataset = ScanObjects(root=self.opt.dataset, partition='test', num_points=self.opt.num_points,few=None)

            self.num_classes = len(dataset.classes)
        
        elif self.opt.dataset_type == 'modelnet40_scanobjects':
            print('(Info) Reading Cross dataset modelnet40,scanobjects')
            few = self.opt.few_shots if self.opt.f else None
            dataset = ModelNet40_ScanObjects(root=self.opt.dataset, partition='train', num_points=self.opt.num_points, few=few
                                     , from_candidates=self.opt.learning_type == 'bCandidate',n_cands=self.opt.n_cands,cands_path=self.opt.cands_path)

            test_dataset = ModelNet40_ScanObjects(root=self.opt.dataset, partition='test', num_points=self.opt.num_points,few=None)

            self.num_classes = len(dataset.classes)

        elif self.opt.dataset_type == 'shapenet':
            print('(Info) Reading shapenet dataset')
            few = self.opt.few_shots if self.opt.f else None
            dataset = ShapeNet(root=self.opt.dataset, partition='train', num_points=self.opt.num_points, few=few
                                     , from_candidates=self.opt.learning_type == 'bCandidate',n_cands=self.opt.n_cands,cands_path=self.opt.cands_path)

            test_dataset = ShapeNet(root=self.opt.dataset, partition='test', num_points=self.opt.num_points,few=None)

            dataset.set_order(self.order)
            test_dataset.set_order(self.order)
            self.num_classes = len(dataset.classes)

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

    def train(self, flag=False, _fe=False, skip=False, lwf=False, stage_id=0):
        ################### Calculate bests ##################
        if stage_id != 0 and self.opt.best_in_stage:
            print(f"\n I'm trying to find bests {self.n_class} of stage : {stage_id}")

            # load model
            prv_n_class = self.n_class-self.opt.step_num_class if stage_id != 0 else self.n_class
            temp_classifier = PointNetCls(k=prv_n_class, feature_transform=False, input_transform=False, last_fc=True, log_softmax=True).copy()

            model_path = f'{self.save_dir}/cls_model_bCandidate_{prv_n_class}.pth' if stage_id != 0 else f'/content/FSCIL/cls/cls_model_bCandidate_{prv_n_class}.pth' 
            temp_classifier.feat.load_state_dict(torch.load(model_path))
            temp_classifier = temp_classifier.cuda()

            # Load dataset
            if self.opt.dataset_type == 'modelnet40':
                temp_dataset = ModelNet40(root=self.opt.dataset, partition='train', num_points=self.opt.num_points,aligned = self.opt.aligned)
            elif self.opt.dataset_type == 'modelnet40_scanobjects':
                temp_dataset = ModelNet40_ScanObjects(root=self.opt.dataset, partition='train', num_points=self.opt.num_points)
            elif self.opt.dataset_type == 'shapenet':
                if stage_id == 8:
                    self.opt.step_num_class -= 1
                    prv_n_class -= 1
                    self.n_class -= 1
                temp_dataset = ShapeNet(root=self.opt.dataset, partition='train', num_points=self.opt.num_points)
            temp_dataset.set_order(self.order)

            # saving folder
            best_save_path = "./temp_samples"
            if os.path.exists(best_save_path) and stage_id == 1:
              shutil.rmtree(best_save_path)

            if not os.path.exists(best_save_path):
                os.mkdir(best_save_path)
            
            self.opt.cands_path = best_save_path


            # make samples
            if self.opt.best_type == 'simple':
                simple_clustring(temp_dataset, temp_classifier, len(self.order), self.class_names, best_save_path, stage_id, prv_n_class, self.opt.n_cands, self.opt.step_num_class, False)
            elif self.opt.best_type == 'disjoint':
                simple_clustring(temp_dataset, temp_classifier, len(self.order), self.class_names, best_save_path, stage_id, prv_n_class, self.opt.n_cands-5, self.opt.step_num_class, True)
            elif self.opt.best_type == 'spectral':
                spectral_clustring_mod(temp_dataset, temp_classifier, len(self.order), self.class_names, best_save_path, stage_id, prv_n_class)


            print("\n")
        #########################################################            


        dataset, test_dataset = self.select_dataset(stage_id)

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
                        cand_ids = np.arange(0, 40*self.opt.n_cands)[:(len(self.classes) - self.opt.step_num_class)*self.opt.n_cands]
                        print('cand_ids: ', cand_ids)

                    dataset.filter(temp, self.except_samples, cand_ids)
                    test_dataset.filter(self.classes)
                else:
                    dataset.filter(self.classes)
                    test_dataset.filter(self.classes)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.opt.batchSize,
            shuffle=True,
            num_workers=int(self.opt.workers))

        testdataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.opt.batchSize,
            shuffle=True,
            num_workers=int(self.opt.workers))

        try:
            (self.save_dir / 'results' if self.opt.test_epoch > 0 else self.save_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass

        if lwf and not flag:
            classifier = PointNetCls(k=self.n_class - self.opt.step_num_class+self.diff,
                                     feature_transform=self.opt.feature_transform, input_transform=self.opt.input_transform, log=True if self.opt.loss_type != 'focal_loss' else False)
            # todo save classifier after first task and save new_model and second task
        else:
            classifier = PointNetCls(k=self.n_class, feature_transform=self.opt.feature_transform, input_transform=self.opt.input_transform, log=True if self.opt.loss_type != 'focal_loss' else False)

        epochs = opt.nepoch
        if _fe and not flag:
            epochs = 120

        if skip:
            print('loading previous model')
            classifier.feat.load_state_dict(
                torch.load('%s/cls_model_%s_%d.pth' % (self.opt.dir_pretrained, self.opt.learning_type, self.n_class)))
        elif _fe and not flag:
            print('loading previous model')
            classifier.feat.load_state_dict(
                torch.load('%s/cls_model_%s_%d.pth' % (
                    self.save_dir, self.opt.learning_type, self.n_class - self.opt.step_num_class+self.diff)))

        if self.opt.model != '':
            classifier.load_state_dict(torch.load(self.opt.model))

        if self.opt.KD and self.opt.learning_type == 'bCandidate':
            old_classifire = classifier.copy().cuda()

        optimizer = optim.Adam(classifier.parameters(), lr=self.opt.lr, betas=(0.9, 0.999))
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
            test_loss[-1], test_acc[-1] = self.test(classifier, testdataloader, False, self.n_class,loss_type=opt.loss_type, stage_id=stage_id)
            self.accuracies.append(test_acc[-1])
            print('%s loss: %f accuracy: %f\n' % (blue('test'), test_loss[-1], test_acc[-1]))
            print()
            torch.save(classifier.feat.state_dict(),
                       '%s/cls_model_%s_%d.pth' % (self.save_dir, self.opt.learning_type, self.n_class))
            return

        if self.opt.loss_type == 'nll_loss':
            classifier.last_fc = True
            classifier.log_softmax = True
            point_loss = PointNetLoss().cuda()
        elif self.opt.loss_type == 'cross_entropy':
            classifier.last_fc = True
            point_loss = torch.nn.CrossEntropyLoss().cuda()
        elif self.opt.loss_type == 'focal_loss':
            classifier.last_fc = True
            classifier.log_softmax = True
            W = torch.FloatTensor([10 if i < self.n_class - self.opt.step_num_class+self.diff else 0.1 for i in range(self.n_class)]).cuda()
            point_loss = FocalLoss(gamma=2, weights=W).cuda()
        else:
            classifier.last_fc = False
            point_loss = AngularPenaltySMLoss(256, self.n_class, loss_type=self.opt.loss_type).cuda()

        if lwf and not flag:
            kd_loss = KnowlegeDistilation(T=float(self.opt.dist_temperature)).cuda()
            shared_model = classifier.copy()
            new_model = PointNetLwf(shared_model, old_k=self.n_class - self.opt.step_num_class+self.diff,
                                    new_k=self.opt.step_num_class).cuda()
            print(new_model)
        
        # If knowledge distillation is selected in bCandicate method
        elif self.opt.KD and self.opt.learning_type == 'bCandidate' and not flag:
            kd_loss = KnowlegeDistilation(T=float(self.opt.dist_temperature)).cuda()

        train_loss = []
        train_acc  = []
        for epoch in range(epochs):
            scheduler.step()
            n = len(dataloader)
            pbar = tqdm(total=n, desc=f'Epoch: {epoch + 1}/{epochs}  ', ncols=110)
            loss_sum = 0
            acc_sum = 0
            for i, data in enumerate(dataloader, 0):
                points, target = data
                points = points.float()
                target = target.type(torch.LongTensor)
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
                    elif self.opt.KD and self.opt.learning_type == 'bCandidate' and not flag:
                        old_vector,_,_ = old_classifire(points)
                        new_vector     = classifier.feature
                        kdl            = kd_loss(new_vector, old_vector)
                        if self.opt.loss_type == 'nll_loss':
                            loss       = point_loss(pred, target, trans_feat, self.opt.feature_transform)
                        elif self.opt.loss_type == 'focal_loss':
                            loss       = point_loss(pred, target)
                        loss          += kdl * self.opt.dist_factor
                        classifier_    = classifier
                    else:
                        classifier_ = classifier
                        if self.opt.loss_type == 'nll_loss':
                            loss = point_loss(pred, target, trans_feat, self.opt.feature_transform)
                        else:
                            loss = point_loss(pred, target)

                    loss.backward()
                    optimizer.step()
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()

                pbar.update(1)
                acc = round(correct.item() / float(self.opt.batchSize), 2)

                acc_sum += acc
                loss_sum += loss.item()
                if lwf and not flag:
                    pbar.postfix = f'loss: {round(loss.item(), 2)}, acc: {acc}, kd_loss: {round(kdl.item(), 2)}'
                else:
                    pbar.postfix = f'loss: {round(loss.item(), 2)}, acc: {acc}'


            pbar.close()

            train_loss.append(loss_sum / n)
            train_acc.append(acc_sum / n)

            test_loss[epoch], test_acc[epoch] = self.test(classifier_, testdataloader, lwf and not flag,
                                                              self.n_class, opt.loss_type, stage_id, last=False)
            if self.opt.test_epoch > 0 and (epoch + 1) % self.opt.test_epoch == 0 or (epoch + 1) == epochs:
                test_loss[epoch], test_acc[epoch] = self.test(classifier_, testdataloader, lwf and not flag,
                                                              self.n_class, opt.loss_type, stage_id, last=True)
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
        torch.save(classifier_.state_dict(),
                    '%s/cls_model_%s_%d_with_head.pth' % (self.save_dir, self.opt.learning_type, self.n_class))


        if _fe:
            self.accuracies.append(test_acc[-1])

        ########################## PLOTTING EVERYTHINGS #############################    
        plt.figure()
        plt.rcParams.update({'font.size': 7})

        plt.subplot(2, 2, 1)
        plt.title('Train Loss')
        plt.plot(train_loss)

        plt.subplot(2, 2, 2)
        plt.title('test Loss')
        plt.plot(test_loss)

        plt.subplot(2, 2, 3)
        plt.title('train acc')
        plt.plot(train_acc)

        plt.subplot(2, 2, 4)
        plt.title('test acc')
        plt.plot(test_acc)

        plt.savefig(f'./results/{stage_id}_plots.png')

    @staticmethod
    def test(classifier, testdataloader, lwf, n_class, loss_type, stage_id, last=False):
        total_loss = 0
        total_correct = 0
        total_testset = 0
        #        if loss_type == 'nll_loss':
        point_loss = PointNetLoss().cuda()
        classifier.last_fc = True
        #       else:
        #          classifier.last_fc = False
        #         point_loss = AngularPenaltySMLoss(256, n_class).cuda()

        pred_list   = [] 
        target_list = []
        for i, data in enumerate(testdataloader, 0):
            points, target = data
            target = target.type(torch.LongTensor)
            points = points.float()
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
            
            loss = F.cross_entropy(pred, target)
            total_loss += loss.item()
            total_correct += correct.item()
            total_testset += points.size()[0]

            # make a list of targets and predictions label for using in confusion-matrix
            for tar,pre in stacked.numpy():
              pred_list.append(pre)
              target_list.append(tar)

        
        # Calculate confusion matrix
        confusion_matrix = metrics.confusion_matrix(target_list, pred_list)
        if last:
          plt.rcParams.update({'font.size': 4})
          cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)
          cm_display.plot()
          plt.savefig(f'./results/CM_{n_class}.png', dpi=400)

        '''
        
        # Data to save as log
        each_class_total = np.sum(confusion_matrix, axis=1) # total data in each class
        each_class_true  = np.diagonal(confusion_matrix)    # total true data in eah class
        # calculate old accuracy and new classes accuracy
        if stage_id != 0:
          old_acc = np.sum(each_class_true[0:20+5*(stage_id-1)]) / np.sum(each_class_total[0:20+5*(stage_id-1)]);old_name = f"0:{20+5*(stage_id-1)}"
          new_acc = np.sum(each_class_true[20+5*(stage_id-1):]) / np.sum(each_class_total[20+5*(stage_id-1):]);new_name = f"{20+5*(stage_id-1)}:{len(each_class_total)}"
        else:
          old_acc = -1;old_name = '0:0'
          new_acc = np.sum(each_class_true) / np.sum(each_class_total);new_name= '0:20'

        accuracy_per_class = {i_c : each_class_true[i_c]/each_class_total[i_c] for i_c in range(len(each_class_total))}


        log_class.data['results'].append({'stage_id': stage_id, 
                                          'accuracy': {
                                              'total'  : total_correct / float(total_testset),
                                              old_name : old_acc,
                                              new_name : new_acc
                                          },
                                          'accuracy_per_class' : accuracy_per_class
                                        })
        
        '''
        return total_loss / float(total_testset), total_correct / float(total_testset)

    def compute_best_samples(self, classifier):
        if self.opt.learning_type == 'bExemplar':
            best_exemplar = Path('utils/best_exemplar.npy')
            if not best_exemplar.exists():
                dataset, _ = self.select_dataset(stage_id = 1)
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
    parser.add_argument(
        '--loss_type', type=str, default='angular', help='')
    parser.add_argument('--lwf', action='store_true', help='is lwf')
    parser.add_argument(
        '--start_num_class', type=int, help='', default=20)
    parser.add_argument(
        '--step_num_class', type=int, help='', default=5)
    parser.add_argument(
        '--n_cands', type=int, help='', default=3)
    parser.add_argument(
        '--manualSeed', type=int, help='', default=1)
    parser.add_argument(
        '--dist_temperature', type=int, default=1, help='distillation temperature')
    parser.add_argument(
        '--dist_factor', type=float, default=0.4, help='distillation factor')
    parser.add_argument(
        '--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--cands_path', type=str, default='cands_path', help='Candidate path')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--f', action='store_true', help='')
    parser.add_argument(
        '--few_shots', type=int, default=5, help='')
    parser.add_argument('--model', type=str, default='', help='model path')
    parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='modelnet40', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")
    parser.add_argument('--input_transform', action='store_true', help="use input transform")
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
    parser.add_argument('--KD', action='store_true', help='')
    parser.add_argument('--order', type=str, default='', help='which ordering will used changed_order, org_order, fscil_order')
    parser.add_argument('--aligned', type=bool, default=False, help='Do you want to use aligned modelnet40 dataset or not')
    parser.add_argument('--best_in_stage', type=bool, default=False, help='Calculate best in files')
    parser.add_argument('--best_type', type=str, default='simple', help='simple | spectral')
    
    opt = parser.parse_args()
    print(opt)

    save_dir = increment_path(Path(opt.outf) / opt.name, exist_ok=opt.exist_ok)

    # opt.manualSeed = random.randint(1, 10000)  # fix seed
    print("Manual Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)

    log_class = Log(opt)
    # log_class.make_json()
    learning = Learning(opt, save_dir)
    learning.start()
    log_class.make_json()