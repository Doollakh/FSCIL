from plyfile import PlyData, PlyElement
import open3d as o3d
import numpy as np
import argparse
import torch
import sys
# Add emd module
sys.path.append("./emd/")
import emd_module as emd


# Read .ply file
def read_ply(filename):
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array

# Input parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '--n_cands', type=int, default=3, help='Number of data on desired folder')
parser.add_argument(
    '--lambda_', type=int, default=0.1, help='default value for lambda')
parser.add_argument(
    '--root', type=str, default='/content/drive/MyDrive/data/data/bExamplr', help='Path to folder')
parser.add_argument(
    '--n_point', type=int, help='number of points', default=2048)
parser.add_argument(
    '--output_path', type=str, default='/content/temp', help='Output path')
parser.add_argument(
    '--total_augment', type=int, help='Total augmentation', default=10)

opt = parser.parse_args()


# Variables
n_cands       = opt.n_cands
root          = opt.root
n_point       = opt.n_point
output_path   = opt.output_path
total_augment = opt.total_augment
lambda_  = opt.lambda_

# An Instance to write point-cloud
pcd = o3d.geometry.PointCloud()

# Class names
classes = ['bed', 'bench', 'bookshelf', 'cup', 'dresser', 'guitar', 'lamp', 'mantel', 'monitor', 'plant', 'radio',
            'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'toilet', 'tv_stand', 'xbox', 'bottle',
            'glass_box', 'night_stand', 'piano', 'vase', 'cone', 'desk', 'door', 'laptop', 'person', 'airplane',
            'bathtub', 'bowl', 'tent', 'wardrobe']
n       = len(classes)

# Reading data and labels from folder
data    = np.zeros(shape=(n * n_cands, n_point, 3))
for j, c in enumerate(classes):
    for i in range(n_cands):
        filename = f'{root}/{c}_{i}.ply'
        plydata = read_ply(filename)
        data[j * n_cands + i, :n_point, :] = plydata #/ np.max(np.abs(plydata))

labels = np.arange(0, n * n_cands) // n_cands

# pass to cuda
points = torch.from_numpy(data).cuda();print(data.shape)
target = torch.from_numpy(labels).cuda();print(labels.shape)

# Save orginal points
count = np.array([0 for i in range(len(classes))])
for idx, label in enumerate(labels):
  pcd.points = o3d.utility.Vector3dVector(points[idx].cpu().detach().numpy())
  o3d.io.write_point_cloud(f"{output_path}/{classes[label]}_{count[label]}.ply", pcd)
  count[label] += 1


# Iterate for each new augmentation
for num in range(n_cands,total_augment):
    print(num)
    # define some parameters
    lam = np.random.beta(1, 1)*lambda_; print("Lamda: ", lam, f"And {int(np.abs(lam*n_point))} will be change")
    B = points.size()[0]

    # shufflig indexs
    rand_index = torch.randperm(B)

    # Targets
    target_a = target.cuda()
    target_b = target[rand_index].cuda()
    
    # Make pair points
    point_a = torch.zeros(B, n_point, 3)
    point_b = torch.zeros(B, n_point, 3)
    point_c = torch.zeros(B, n_point, 3)
    point_a = points.cuda()
    point_b = points[rand_index].cuda()
    point_c = points[rand_index].cuda()
    
    # Running EMD 
    remd = emd.emdModule()
    remd = remd.cuda()
    dis, ind = remd(point_a, point_b, 0.005, 300)
    for ass in range(B):
        point_c[ass, :, :] = point_c[ass, ind[ass].long(), :]

    # Total point to be changes
    int_lam = int(n_point * lam)
    int_lam = max(1, int_lam)

    # KNN method to replacing
    random_point = torch.from_numpy(np.random.choice(1024, B, replace=False, p=None))
    ind1 = torch.tensor(range(B))
    query = point_a[ind1, random_point].view(B, 1, 3)
    dist = torch.sqrt(torch.sum((point_a - query.repeat(1, n_point, 1)) ** 2, 2))
    idxs = dist.topk(int_lam, dim=1, largest=False, sorted=True).indices
    for i2 in range(B):
        points[i2, idxs[i2], :] = point_c[i2, idxs[i2], :]

    for idx, point in enumerate(points):
        pcd.points = o3d.utility.Vector3dVector(point.cpu().detach().numpy())
        o3d.io.write_point_cloud(f"{output_path}/{classes[labels[idx]]}_{num}.ply", pcd)





    
