# simple clustring
import tqdm
import numpy as np
from sklearn.cluster import KMeans
from plyfile import PlyData, PlyElement
import open3d as o3d
import os
import torch

def simple_clustring(dataset, classifier, n_class, classes_name, save_path, stage_id, start_class):
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
      pred, _, _, _ = classifier.feat(points)

      X = np.concatenate((X, pred.cpu().detach().numpy()), 0) if i != 0 else pred.cpu().detach().numpy()
      y = np.concatenate((y, target.cpu().detach().numpy()), 0) if i != 0 else target.cpu().detach().numpy()

  cos = lambda in1, in2: np.dot(in1, in2) / (np.linalg.norm(in1) * np.linalg.norm(in2))

  uniq = np.unique(y, return_counts=True)[1]
  result = {i:[] for i in range(n_class)}

  which_classes = range(n_class) if stage_id == 1 else range(start_class,n_class )
  for i in which_classes:
      w = np.where(y == i)[0]
      cX = X[w]

      #clustring
      print(f'class {i}:{classes_name[i]}')
      kmeans = KMeans(n_clusters=5, n_init='auto').fit(cX)
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

  # Save as best exampler
  for i in range(n_class):
    id_list_of_top = result[i]
    for j in range(len(id_list_of_top)):
      pcd.points = o3d.utility.Vector3dVector(dataset[id_list_of_top[j]][0])
      o3d.io.write_point_cloud(os.path.join(save_path,f"{classes_name[i]}_{j}.ply"), pcd)

