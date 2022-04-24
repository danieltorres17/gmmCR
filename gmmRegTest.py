import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

import copy

def main():
  # load point cloud
  path = "data/region_growing_tutorial.pcd"
  # path = "../Edge_Extraction/ArtificialPointClouds/SacModel.pcd"
  pcd = o3d.io.read_point_cloud(path, format="pcd")
  o3d.visualization.draw_geometries([pcd]) # visualization

  # set source point cloud
  source_pc = pcd
  # set target point cloud
  target_pc = copy.deepcopy(source_pcd)



if __name__ == "__main__":
  # main()

  a = np.arange(start=1, stop=10)
  print(a)

