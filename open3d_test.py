import open3d as o3d
import numpy as np
from gmm import GMM 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import viz_tools as vt

def main():
  # path = "data/region_growing_tutorial.pcd"
  # path = "../Edge_Extraction/ArtificialPointClouds/SacModel.pcd"
  path = "cube_eges.pcd"
  pcd = o3d.io.read_point_cloud(path, format="pcd")
  # o3d.visualization.draw_geometries([pcd])
  # o3d.io.write_point_cloud("cube_eges.pcd", pcd)

  # store edges point cloud as numpy array
  X = np.asarray(pcd.points)
  # print(f"edges_pc shape: {edges_pc.shape}")
  # print(f"edges_pc max: {np.max(edges_pc, axis=0)}")
  # max_pt = pcd.get_max_bound()
  # print(f"max bound: {max_pt}")
  # min_pt = pcd.get_min_bound()
  # print(f"min bound: {min_pt}")
  gmm = GMM(X)
  gmm.fit(X)
  print("custom GMM estimated means: ")
  print(gmm.cluster_means)
  print()
  print("custom GMM estimated covariances: ")
  print(gmm.cluster_covs)
  print()
  print("custom GMM estimated weights: ")
  print(gmm.weights)
  print()
  print()

  # plot test data
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(X[:,0],X[:,1],X[:,2])
  # plt.grid()
  for l in range(len(gmm.cluster_means)):
    ax.scatter(gmm.cluster_means[l][0], gmm.cluster_means[l][1], gmm.cluster_means[l][2], s=75)
    # ax.scatter(gm_test.means_[l][0], gm_test.means_[l][1], gm_test.means_[l][2], marker='x', s=75)
  
  # set labels
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  plt.show()

if __name__ == "__main__":
  main()
