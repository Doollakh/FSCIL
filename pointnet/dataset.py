from __future__ import print_function
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import sys
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement
import glob
import h5py


def read_ply(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def read_candidates(root, n_cands=3, dataset_type='modelnet40'):
    if dataset_type == 'modelnet40':
        classes = list(np.load('./misc/class_names.npy'))
    elif dataset_type == 'scanobjects':
        classes = ['bag','bin','box','bed','chair','desk','display','door','shelves','table','cabinets','pillow','sink','sofa','toilet']
    elif dataset_type == 'modelnet40_scanobjects':
        classes = ['airplane','bathtub','bottle','bowl','car','cone','cup','curtain','flower pot','glass box','guitar','keyboard','lamp','laptop','mantel','night stand','person','piano','plant','radio','range hood','stairs','tent','tv stand','vase','cabinet','chair','desk','display','door','shelf','table','bed','sink','sofa','toilet']
    elif dataset_type == 'shapenet':
        classes = ['airplane', 'bag', 'basket' , 'bathtub' , 'bed' , 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellphone', 'chair', 'clock', 'dishwasher', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'keyboard', 'knife', 'lamp', 'laptop', 'mailbox', 'microphone', 'microwave', 'monitor', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket', 'skateboard', 'sofa', 'speaker', 'stove', 'table', 'telephone', 'tin_can', 'tower', 'train', 'vessel', 'washer']

    n = len(classes)
    data = np.zeros(shape=(n * n_cands, 1024, 3))
    for j, c in enumerate(classes):
        for i in range(n_cands):
            try:
              filename = f'{root}/{c}_{i}.ply'
              # read filename.ply to numpy array
              plydata = read_ply(filename)
              data[j * n_cands + i, :1024, :] = plydata #/ np.max(np.abs(plydata))
            except:
              pass
            #   print(f"{c}_{i} not found but I think It's not important")

    return data, np.arange(0, n * n_cands) // n_cands


def get_segmentation_classes(root):
    catfile = os.path.join(root, 'synsetoffset2category.txt')
    cat = {}
    meta = {}

    with open(catfile, 'r') as f:
        for line in f:
            ls = line.strip().split()
            cat[ls[0]] = ls[1]

    for item in cat:
        dir_seg = os.path.join(root, cat[item], 'points_label')
        dir_point = os.path.join(root, cat[item], 'points')
        fns = sorted(os.listdir(dir_point))
        meta[item] = []
        for fn in fns:
            token = (os.path.splitext(os.path.basename(fn))[0])
            meta[item].append((os.path.join(dir_point, token + '.pts'), os.path.join(dir_seg, token + '.seg')))

    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'C:/Users/Sajjad/Documents/py/Untitled Folder 1/pointnet.pytorch/misc/num_seg_classes.txt'),
              'w') as f:
        for item in cat:
            datapath = []
            num_seg_classes = 0
            for fn in meta[item]:
                datapath.append((item, fn[0], fn[1]))

            for i in tqdm(range(len(datapath))):
                l = len(np.unique(np.loadtxt(datapath[i][-1]).astype(np.uint8)))
                if l > num_seg_classes:
                    num_seg_classes = l

            print("category {} num segmentation classes {}".format(item, num_seg_classes))
            f.write("{}\t{}\n".format(item, num_seg_classes))


def gen_modelnet_id(root):
    classes = []
    with open(os.path.join(root, 'train.txt'), 'r') as f:
        for line in f:
            classes.append(line.strip().split('/')[0])
    classes = np.unique(classes)
    with open(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                           'C:/Users/Sajjad/Documents/py/Untitled Folder 1/pointnet.pytorch/misc/modelnet_id.txt'),
              'w') as f:
        for i in range(len(classes)):
            f.write('{}\t{}\n'.format(classes[i], i))


class ShapeNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 classification=False,
                 class_choice=None,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
        self.cat = {}
        self.data_augmentation = data_augmentation
        self.classification = classification
        self.seg_classes = {}

        with open(self.catfile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = ls[1]
        # print(self.cat)
        if not class_choice is None:
            self.cat = {k: v for k, v in self.cat.items() if k in class_choice}

        self.id2cat = {v: k for k, v in self.cat.items()}

        self.meta = {}
        splitfile = os.path.join(self.root, 'train_test_split', 'shuffled_{}_file_list.json'.format(split))
        # from IPython import embed; embed()
        filelist = json.load(open(splitfile, 'r'))
        for item in self.cat:
            self.meta[item] = []

        for file in filelist:
            _, category, uuid = file.split('/')
            if category in self.cat.values():
                self.meta[self.id2cat[category]].append((os.path.join(self.root, category, 'points', uuid + '.pts'),
                                                         os.path.join(self.root, category, 'points_label',
                                                                      uuid + '.seg')))

        self.datapath = []
        for item in self.cat:
            for fn in self.meta[item]:
                self.datapath.append((item, fn[0], fn[1]))

        self.classes = dict(zip(sorted(self.cat), range(len(self.cat))))
        print(self.classes)
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/num_seg_classes.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.seg_classes[ls[0]] = int(ls[1])
        self.num_seg_classes = self.seg_classes[list(self.cat.keys())[0]]
        print(self.seg_classes, self.num_seg_classes)

    def __getitem__(self, index):
        fn = self.datapath[index]
        cls = self.classes[self.datapath[index][0]]
        point_set = np.loadtxt(fn[1]).astype(np.float32)
        seg = np.loadtxt(fn[2]).astype(np.int64)
        # print(point_set.shape, seg.shape)

        choice = np.random.choice(len(seg), self.npoints, replace=True)
        # resample
        point_set = point_set[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        seg = seg[choice]
        point_set = torch.from_numpy(point_set)
        seg = torch.from_numpy(seg)
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.classification:
            return point_set, cls
        else:
            return point_set, seg

    def __len__(self):
        return len(self.datapath)


class ModelNetDataset(data.Dataset):
    def __init__(self,
                 root,
                 npoints=2500,
                 split='train',
                 data_augmentation=True):
        self.npoints = npoints
        self.root = root
        self.split = split
        self.data_augmentation = data_augmentation
        self.fns = []
        with open(os.path.join(root, '{}.txt'.format(self.split)), 'r') as f:
            for line in f:
                self.fns.append(line.strip())

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '/misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        print(self.cat)
        self.classes = list(self.cat.keys())

    def __getitem__(self, index):
        fn = self.fns[index]
        cls = self.cat[fn.split('/')[0]]
        with open(os.path.join(self.root, fn), 'rb') as f:
            plydata = PlyData.read(f)
        pts = np.vstack([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T
        choice = np.random.choice(len(pts), self.npoints, replace=True)
        point_set = pts[choice, :]

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # scale

        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # random rotation
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # random jitter

        point_set = torch.from_numpy(point_set.astype(np.float32))
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))
        return point_set, cls

    def __len__(self):
        return len(self.fns)


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def download(root):
    if not os.path.exists(root):
        os.mkdir(root)
    if not os.path.exists(os.path.join(root, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], root))
        os.system('rm %s' % (zipfile))


def load_data(root, partition):
    root = os.path.join(root, 'data')
    download(root)
    all_data = []
    all_label = []
    g = sorted(glob.glob(os.path.join(root, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)))
    for h5_name in g:
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
        # print(all_data)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40(data.Dataset):
    def __init__(self, root, num_points, partition='train', few=None, from_candidates=False,n_cands=3,cands_path='/content', aligned = False):

        self.memory_candidates = None
        if from_candidates:
            self.memory_candidates = read_candidates(cands_path,n_cands)

        if aligned:
            if partition == 'train':
                root_data = os.path.join(root, 'modelnet40_aligned','train_data.h5')
            elif partition == 'test':
                root_data = os.path.join(root, 'modelnet40_aligned','test_data.h5')

            # Reading aligned data
            f = h5py.File(root_data)
            self.data  = f['data'][:].astype('float32')
            self.label = f['label'][:].astype('int8')

        else:
            self.data, self.label = load_data(root, partition)
        self.num_points = num_points
        self.partition = partition

        self.data = self.data[:,:self.num_points,:]

        self.cat = {}
        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../misc/modelnet_id.txt'), 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cat[ls[0]] = int(ls[1])

        self.classes = list(self.cat.keys())

        if partition == 'train' and few is not None:
            ids = []
            c = np.zeros(40)
            for i, j in enumerate(self.label):
                if c[j] < few:
                    ids.append(i)
                    c[j] += 1

            self.data = self.data[ids]
            self.label = self.label[ids]
            print(np.unique(self.label, return_counts=True))

    def filter(self, classes, except_samples=None, cand_ids=None):
        if except_samples is None:
            except_samples = []
        f = [i for i, item in enumerate(self.label) if (item in classes) or (i in except_samples)]
        self.label = self.label[f]
        self.data = self.data[f]

        if self.memory_candidates is not None:
            if cand_ids is not None:
                self.data = np.append(self.data, self.memory_candidates[0][cand_ids], axis=0)
                self.label = np.append(self.label, self.memory_candidates[1][cand_ids])

        self.classes = [c for i, c in enumerate(self.classes) if i in classes]
        print(self.classes)
    
    def normalize(self):
        temp_data = np.zeros_like(self.data)
        for idx, sample_data in enumerate(self.data):
            temp_data[idx,:,0] = sample_data[:,0] / np.max(np.abs(sample_data[:,0]))
            temp_data[idx,:,1] = sample_data[:,1] / np.max(np.abs(sample_data[:,1]))
            temp_data[idx,:,2] = sample_data[:,2] / np.max(np.abs(sample_data[:,2]))

        self.data = temp_data

    def set_order(self, order):
        self.classes = [self.classes[i] for i in order]
        self.label = self._map_new_class_index(self.label, order).reshape(-1)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: np.where(order == x), y)), dtype=np.int64)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
#        if self.partition == 'train':
#            pointcloud = translate_pointcloud(pointcloud)
#            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ScanObjects(data.Dataset):
    def __init__(self, root, num_points, partition='train', few=None, from_candidates=False,n_cands=3,cands_path='/content'):
        # Read Candidates
        self.memory_candidates = None
        if from_candidates:
            try:
                self.memory_candidates = read_candidates(cands_path,n_cands,dataset_type='scanpbjects')
            except:
                print('(NOTE)  Loading bests is not completed for this stage!')
        # Load Scan objects data from h5 file
        if partition == 'train':
            root = os.path.join(root, 'scan_objects','training_objectdataset_nobg.h5')
        elif partition == 'test':
            root = os.path.join(root, 'scan_objects','test_objectdataset_nobg.h5')
        f = h5py.File(root)

        self.data  = f['data'][:].astype('float32')
        self.label = f['label'][:].astype('int8')

        self.num_points = num_points
        self.partition = partition

        self.classes = ['bag','bin','box','bed','chair','desk','display','door','shelves','table','cabinets','pillow','sink','sofa','toilet']

        if partition == 'train' and few is not None:
            ids = []
            c = np.zeros(15)
            for i, j in enumerate(self.label):
                if c[j] < few:
                    ids.append(i)
                    c[j] += 1

            self.data = self.data[ids]
            self.label = self.label[ids]
            print(np.unique(self.label, return_counts=True))

    def filter(self, classes, except_samples=None, cand_ids=None):
        if except_samples is None:
            except_samples = []
        f = [i for i, item in enumerate(self.label) if (item in classes) or (i in except_samples)]
        self.label = self.label[f]
        self.data = self.data[f]

        if self.memory_candidates is not None:
            if cand_ids is not None:
                self.data = np.append(self.data, self.memory_candidates[0][cand_ids], axis=0)
                self.label = np.append(self.label, self.memory_candidates[1][cand_ids])

        self.classes = [c for i, c in enumerate(self.classes) if i in classes]
        print(self.classes)
    
    def normalize(self):
        temp_data = np.zeros_like(self.data)
        for idx, sample_data in enumerate(self.data):
            temp_data[idx,:,0] = sample_data[:,0] / np.max(np.abs(sample_data[:,0]))
            temp_data[idx,:,1] = sample_data[:,1] / np.max(np.abs(sample_data[:,1]))
            temp_data[idx,:,2] = sample_data[:,2] / np.max(np.abs(sample_data[:,2]))

        self.data = temp_data

    def set_order(self, order):
        self.classes = [self.classes[i] for i in order]
        self.label = self._map_new_class_index(self.label, order).reshape(-1)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: np.where(order == x), y)), dtype=np.int64)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
#        if self.partition == 'train':
#            pointcloud = translate_pointcloud(pointcloud)
#            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40_ScanObjects(data.Dataset):
    def __init__(self, root, num_points, partition='train', few=None, from_candidates=False,n_cands=3,cands_path='/content'):
        # Read Candidates
        self.memory_candidates = None
        if from_candidates:
            self.memory_candidates = read_candidates(cands_path,n_cands,dataset_type='modelnet40_scanobjects')

        # Load Scan objects data from h5 file
        if partition == 'train':
            root_data = os.path.join(root, 'modelnet_scanobject','train_data.npy')
            root_label = os.path.join(root, 'modelnet_scanobject','train_label.npy')
        elif partition == 'test':
            root_data = os.path.join(root, 'modelnet_scanobject','test_data.npy')
            root_label = os.path.join(root, 'modelnet_scanobject','test_label.npy')

        self.data  = np.load(root_data)
        self.label = np.load(root_label)

        self.num_points = num_points
        self.partition  = partition

        self.classes = ['airplane','bathhub','bottle','bowl','car','cone','cup','curtain','flower pot','glass box','guitar','keyboard','lamp','laptop','mantel','night stand','person','piano','plant','radio','range hood','stairs','tent','tv stand','vase','cabinet','chair','desk','display','door','shelf','table','bed','sink','sofa','toilet']

        if partition == 'train' and few is not None:
            ids = []
            c = np.zeros(len(self.classes))
            for i, j in enumerate(self.label):
                if c[j] < few:
                    ids.append(i)
                    c[j] += 1

            self.data = self.data[ids]
            self.label = self.label[ids]
            print(np.unique(self.label, return_counts=True))

    def filter(self, classes, except_samples=None, cand_ids=None):
        if except_samples is None:
            except_samples = []
        f = [i for i, item in enumerate(self.label) if (item in classes) or (i in except_samples)]
        self.label = self.label[f]
        self.data = self.data[f]

        if self.memory_candidates is not None:
            if cand_ids is not None:
                self.data = np.append(self.data, self.memory_candidates[0][cand_ids], axis=0)
                self.label = np.append(self.label, self.memory_candidates[1][cand_ids])

        self.classes = [c for i, c in enumerate(self.classes) if i in classes]
        print(self.classes)
    
    def normalize(self):
        temp_data = np.zeros_like(self.data)
        for idx, sample_data in enumerate(self.data):
            temp_data[idx,:,0] = sample_data[:,0] / np.max(np.abs(sample_data[:,0]))
            temp_data[idx,:,1] = sample_data[:,1] / np.max(np.abs(sample_data[:,1]))
            temp_data[idx,:,2] = sample_data[:,2] / np.max(np.abs(sample_data[:,2]))

        self.data = temp_data

    def set_order(self, order):
        self.classes = [self.classes[i] for i in order]
        self.label = self._map_new_class_index(self.label, order).reshape(-1)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: np.where(order == x), y)), dtype=np.int64)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
#        if self.partition == 'train':
#            pointcloud = translate_pointcloud(pointcloud)
#            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]



class ShapeNet(data.Dataset):
    def __init__(self, root, num_points, partition='train', few=None, from_candidates=False,n_cands=3,cands_path='/content'):
        # Read Candidates
        self.memory_candidates = None
        if from_candidates:
            self.memory_candidates = read_candidates(cands_path,n_cands,dataset_type='shapenet')

        # Load Scan objects data from h5 file
        if partition == 'train':
            root_data = os.path.join(root, 'shapenet','train_data_2.npy')
            root_label = os.path.join(root, 'shapenet','train_label_2.npy')
        elif partition == 'test':
            root_data = os.path.join(root, 'shapenet','test_data_2.npy')
            root_label = os.path.join(root, 'shapenet','test_label_2.npy')

        self.data  = np.load(root_data, allow_pickle=True)
        self.label = np.load(root_label, allow_pickle=True)

        self.num_points = num_points
        self.partition  = partition

        self.classes = ['airplane', 'bag', 'basket' , 'bathtub' , 'bed' , 'bench', 'birdhouse', 'bookshelf', 'bottle', 'bowl', 'bus', 'cabinet', 'camera', 'can', 'cap', 'car', 'cellphone', 'chair', 'clock', 'dishwasher', 'earphone', 'faucet', 'file', 'guitar', 'helmet', 'jar', 'keyboard', 'knife', 'lamp', 'laptop', 'mailbox', 'microphone', 'microwave', 'monitor', 'motorcycle', 'mug', 'piano', 'pillow', 'pistol', 'pot', 'printer', 'remote_control', 'rifle', 'rocket', 'skateboard', 'sofa', 'speaker', 'stove', 'table', 'telephone', 'tin_can', 'tower', 'train', 'vessel', 'washer']

        if partition == 'train' and few is not None:
            ids = []
            c = np.zeros(len(self.classes))
            for i, j in enumerate(self.label):
                if c[j] < few:
                    ids.append(i)
                    c[j] += 1

            self.data = self.data[ids]
            self.label = self.label[ids]
            print(np.unique(self.label, return_counts=True))

    def filter(self, classes, except_samples=None, cand_ids=None):
        if except_samples is None:
            except_samples = []
        f = [i for i, item in enumerate(self.label) if (item in classes) or (i in except_samples)]
        self.label = self.label[f]
        self.data = self.data[f]

        if self.memory_candidates is not None:
            if cand_ids is not None:
                self.data = np.append(self.data, self.memory_candidates[0][cand_ids], axis=0)
                self.label = np.append(self.label, self.memory_candidates[1][cand_ids])

        self.classes = [c for i, c in enumerate(self.classes) if i in classes]
        print(self.classes)
    
    def normalize(self):
        temp_data = np.zeros_like(self.data)
        for idx, sample_data in enumerate(self.data):
            temp_data[idx,:,0] = sample_data[:,0] / np.max(np.abs(sample_data[:,0]))
            temp_data[idx,:,1] = sample_data[:,1] / np.max(np.abs(sample_data[:,1]))
            temp_data[idx,:,2] = sample_data[:,2] / np.max(np.abs(sample_data[:,2]))

        self.data = temp_data

    def set_order(self, order):
        self.classes = [self.classes[i] for i in order]
        self.label = self._map_new_class_index(self.label, order).reshape(-1)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: np.where(order == x), y)), dtype=np.int64)

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
#        if self.partition == 'train':
#            pointcloud = translate_pointcloud(pointcloud)
#            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    dataset = sys.argv[1]
    datapath = sys.argv[2]

    if dataset == 'shapenet':
        d = ShapeNetDataset(root=datapath, class_choice=['Chair'])
        print(len(d))
        ps, seg = d[0]
        print(ps.size(), ps.type(), seg.size(), seg.type())

        d = ShapeNetDataset(root=datapath, classification=True)
        print(len(d))
        ps, cls = d[0]
        print(ps.size(), ps.type(), cls.size(), cls.type())
        # get_segmentation_classes(datapath)

    if dataset == 'modelnet':
        gen_modelnet_id(datapath)
        d = ModelNetDataset(root=datapath)
        print(len(d))
        print(d[0])
