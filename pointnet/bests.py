# simple clustring
import tqdm
import numpy as np
from sklearn.cluster import KMeans
from plyfile import PlyData, PlyElement
import open3d as o3d
import os
import torch
import pickle
import random
import shutil
from sklearn.cluster import SpectralClustering
from scipy.spatial.distance import cosine
from scipy.linalg import eigh
import numpy as np
import scipy.spatial
import scipy

from pointnet.specteral import SpectralClusteringMod


cos = lambda in1, in2: np.dot(in1, in2) / (np.linalg.norm(in1) * np.linalg.norm(in2))


def simple_clustring(dataset, classifier, n_class, classes_name, save_path, stage_id, start_class, number_of_output, steps, disjoint=False):
  dataloader = torch.utils.data.DataLoader(
                  dataset,
                  batch_size=32,
                  shuffle=False,
                  num_workers=int(2))
  labels_dict = {}
  feature_dict = {}
  print('computing best samples...')
  for i, data in enumerate(dataloader, 0):
      points, target = data
      points = points.transpose(2, 1)
      points, target = points.cuda(), target.cuda()
      classifier = classifier.eval()
      _, _, _, _,pred = classifier.feat(points)

      X = np.concatenate((X, pred.cpu().detach().numpy()), 0) if i != 0 else pred.cpu().detach().numpy()
      y = np.concatenate((y, target.cpu().detach().numpy()), 0) if i != 0 else target.cpu().detach().numpy()

  cos = lambda in1, in2: np.dot(in1, in2) / (np.linalg.norm(in1) * np.linalg.norm(in2))

  uniq = np.unique(y, return_counts=True)[1]
  result = {i:[] for i in range(n_class)}

  which_classes = range(start_class) if stage_id == 1 else range(start_class-steps,start_class)
  for i in which_classes:
      w = np.where(y == i)[0]
      cX = X[w]

      #clustring
      print(f'class {i}:{classes_name[i]}')
      kmeans = KMeans(n_clusters=number_of_output, n_init='auto').fit(cX)
      labels_dict[classes_name[i]] = kmeans.labels_

      # select groups
      for c_label in range(np.max(kmeans.labels_)+1):
        selected_X = cX[kmeans.labels_ == c_label]
        selected_w = w[kmeans.labels_ == c_label]
        n = len(selected_X)
        r = np.zeros(n)
        avg = np.average(selected_X, axis=0)
        for j, item in enumerate(selected_X):
            r[j] = cos(avg, item)
        args = np.argsort(r)[::-1][:n]
        result[i].append(selected_w[args[0]])

  # Save samples
  pcd = o3d.geometry.PointCloud()

  # selected sample
  if disjoint: selected_sample = np.load('/content/drive/MyDrive/data/data/input_sample_5.npy', allow_pickle=True)[0]
 
  # Save as best exampler
  for i in which_classes:
    if disjoint : result[i] += selected_sample[classes_name[i]]
    id_list_of_top = result[i]
    for j in range(len(id_list_of_top)):
      pcd.points = o3d.utility.Vector3dVector(dataset[id_list_of_top[j]][0])
      o3d.io.write_point_cloud(os.path.join(save_path,f"{classes_name[i]}_{j}.ply"), pcd)


def chamfer_distance_modified(point_cloud_A, point_cloud_B, feature_points_A, feature_points_B):
    # Build k-d trees for fast nearest neighbor search
    tree_A = scipy.spatial.cKDTree(point_cloud_A)
    tree_B = scipy.spatial.cKDTree(point_cloud_B)

    # Compute the distances from A to B and B to A
    dist_A_to_B, nn_inds_A_to_B = tree_A.query(point_cloud_B, k=1)
    dist_B_to_A, nn_inds_B_to_A = tree_B.query(point_cloud_A, k=1)

    term1 = np.sum((feature_points_A[nn_inds_A_to_B] - feature_points_B)**2)
    term2 = np.sum((feature_points_B[nn_inds_B_to_A] - feature_points_A)**2)
    
    
    # Return the sum of the two terms
    return term1 + term2


def spectral_clustring(dataset, classifier, n_class, classes_name, save_path, stage_id, start_class):
  dataloader = torch.utils.data.DataLoader(
                  dataset,
                  batch_size=32,
                  shuffle=False,
                  num_workers=int(2))

  print('computing best samples...')

  result = {i:[] for i in range(n_class)}
  which_classes = range(start_class) if stage_id == 1 else range(start_class-5,start_class)
  for class_idx in which_classes:
    print(f'class {class_idx}:{classes_name[class_idx]}')
    X = np.array([])
    y = np.array([])
    P = np.array([])
    for data in dataloader:
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        target = target.cpu().detach().numpy()

        ##### checking and selecting based on each classe
        which = target == np.array([class_idx for _ in range(len(target))])
        if sum(which) != 0:
          target = target[which]
          points = points[which]

          classifier = classifier.eval()
          ـ , ـ, ـ, pred = classifier.feat(points)

          X = np.concatenate((X, pred.detach().cpu().numpy()), 0) if len(X) != 0 else pred.cpu().detach().numpy()
          y = np.concatenate((y, target), 0) if len(y) != 0 else target
          P = np.concatenate((P, points.permute((0,2,1)).detach().cpu().numpy()), 0) if len(P) != 0 else points.permute((0,2,1)).cpu().detach().numpy()
        
          # break


    dis_matrix = np.zeros((len(X),len(X)))

    for b in range(len(X)):
        for c in range(len(X)):
          dis_matrix[b,c] =  chamfer_distance_modified(P[b], P[c], X[b], X[c])
    # break
    clustering = SpectralClustering(n_clusters=5, random_state=0, n_init=500).fit(dis_matrix)

    # select groups
    for c_label in range(np.max(clustering.labels_)+1):
      selected_X = X[clustering.labels_ == c_label]
      selected_w = y[clustering.labels_ == c_label]
      n = len(selected_X)
      r = np.zeros(n)
      avg = np.average(selected_X, axis=0)
      for j, item in enumerate(selected_X):
          r[j] = cos(avg, item)
      args = np.argsort(r)[::-1][:n]
      result[class_idx].append(selected_w[args[0]])

    # Save samples
    pcd = o3d.geometry.PointCloud()

    # Save as best exampler
    for i in which_classes:
      id_list_of_top = result[i]
      for j in range(len(id_list_of_top)):
        pcd.points = o3d.utility.Vector3dVector(dataset[id_list_of_top[j]][0])
        o3d.io.write_point_cloud(os.path.join(save_path,f"{classes_name[i]}_{j}.ply"), pcd)



def spectral_clustring_mod(dataset, classifier, n_class, classes_name, save_path, stage_id, start_class):
  dataloader = torch.utils.data.DataLoader(
                  dataset,
                  batch_size=32,
                  shuffle=False,
                  num_workers=int(2))

  print('computing best samples...')

  result = {i:[] for i in range(n_class)}

  which_classes = range(start_class) if stage_id == 1 else range(start_class-5,start_class)
  for class_idx in which_classes:
    print(f'class {class_idx}:{classes_name[class_idx]}')
    X = np.array([])
    y = np.array([])
    P = np.array([])
    for data in dataloader:
        points, target = data
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        target = target.cpu().detach().numpy()

        ##### checking and selecting based on each classe
        which = target == np.array([class_idx for _ in range(len(target))])
        if sum(which) != 0:
          target = target[which]
          points = points[which]

          classifier = classifier.eval()
          ـ , ـ, ـ, pred, _ = classifier.feat(points)

          X = np.concatenate((X, pred.detach().cpu().numpy()), 0) if len(X) != 0 else pred.cpu().detach().numpy()
          y = np.concatenate((y, target), 0) if len(y) != 0 else target
          P = np.concatenate((P, points.permute((0,2,1)).detach().cpu().numpy()), 0) if len(P) != 0 else points.permute((0,2,1)).cpu().detach().numpy()
        
          # break


    dis_matrix = np.zeros((len(X),len(X)))
    affinity_matrix = np.zeros((len(X),len(X)))

    for b in range(len(X)):
        for c in range(len(X)):
          dis_matrix[b,c] =  chamfer_distance_modified(P[b], P[c], X[b], X[c])
    
    sigma = int(np.median(dis_matrix[dis_matrix != 0]))

    for b in range(len(X)):
        for c in range(len(X)):
          distance = chamfer_distance_modified(P[b], P[c], X[b], X[c])
          affinity_matrix[b,c] =  np.exp(-distance**2 / (2.0 * sigma**2))

    # break
    clustering = SpectralClusteringMod(n_clusters=5, random_state=0,affinity='precomputed').fit(affinity_matrix)

    # select groups
    for c_label in range(np.max(clustering.labels_)+1):
      selected_X = X[clustering.labels_ == c_label]
      selected_w = y[clustering.labels_ == c_label]
      n = len(selected_X)
      r = np.zeros(n)
      avg = np.average(selected_X, axis=0)
      for j, item in enumerate(selected_X):
          r[j] = cos(avg, item)
      args = np.argsort(r)[::-1][:n]
      result[class_idx].append(selected_w[args[0]])

    # Save samples
    pcd = o3d.geometry.PointCloud()

    # Save as best exampler
    for i in which_classes:
      id_list_of_top = result[i]
      for j in range(len(id_list_of_top)):
        pcd.points = o3d.utility.Vector3dVector(dataset[id_list_of_top[j]][0])
        o3d.io.write_point_cloud(os.path.join(save_path,f"{classes_name[i]}_{j}.ply"), pcd)

