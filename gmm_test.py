import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from utils import *
import open3d as o3d
import time

from gmm import GMM

def main():
  np.set_printoptions(precision=3)
  
  gen_dataset = True

  if not gen_dataset:
    """
    Generate toy datasets
    """
    # number of clusters
    num_clusters = 5
    # get data
    X, y = make_blobs(n_samples=100, centers=num_clusters, n_features=3, cluster_std=0.5)

  else:
    """
    Run GMM on point cloud datasets
    """
    path = "sphereMultiple_edges.pcd"
    pcd = o3d.io.read_point_cloud(path, format="pcd")
    X = np.asarray(pcd.points, dtype=np.float64)

  # get GMM object and fit on data
  start = time.time()
  gmm = GMM(X)
  gmm.fit(X)
  end = time.time() - start
  print(end)
  print("custom GMM estimated means: ")
  gmm_means = gmm.cluster_means
  print(gmm_means)
  print(bmatrix(gmm_means))
  print()
  print()
  print("custom GMM estimated covariances: ")
  gmm_covs = gmm.cluster_covs
  print(gmm_covs)
  gmm_covs = [c[0][0] for c in gmm_covs]
  print(bmatrix(np.array(gmm_covs), True))
  print()
  print("custom GMM estimated weights: ")
  gmm_weights = gmm.weights
  print(gmm_weights)
  print(bmatrix(np.array(gmm_weights), True))
  print()
  print()

  # save generated GMM parameters to file
  np.save('gmm_multiple_sphere_means.npy', gmm.cluster_means)
  np.save('gmm_multiple_sphere_covs.npy', gmm.cluster_covs)
  np.save('gmm_multiple_sphere_weights.npy', gmm.weights)

  # comparison model
  # gm_test = GaussianMixture(n_components=num_clusters, covariance_type='spherical').fit(X)
  # # gm_test = GaussianMixture(n_components=num_clusters).fit(X)
  # print("sklearn GMM estimated means: ")
  # print(gm_test.means_)
  # print(bmatrix(gm_test.means_))
  # print()
  # print("sklearn GMM estimated covariances: ")
  # print(gm_test.covariances_)
  # print(bmatrix(gm_test.covariances_, True))
  # print()
  # print("sklearn GMM estimated weights: ")
  # print(gm_test.weights_)
  # print(bmatrix(gm_test.weights_, True))

  # save generated GMM parameters to file
  # np.save('gmm_means.npy', gmm.cluster_means)
  # np.save('gmm_covs.npy', gmm.cluster_covs)
  # np.save('gmm_weights.npy', gmm.weights)

  # plot test data
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(X[:,0],X[:,1],X[:,2])

  for l in range(len(gmm.cluster_means)):
    ax.scatter(gmm.cluster_means[l][0], gmm.cluster_means[l][1], gmm.cluster_means[l][2], s=300)
    # ax.scatter(gm_test.means_[l][0], gm_test.means_[l][1], gm_test.means_[l][2], marker='x', s=100)
  
  # set labels
  # ax.set_title("Num Clusters: 5")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  ax.axis('tight')

  # plt.savefig("sphere_edges_gmm_test.png")
  plt.show()

if __name__ == "__main__":
  main()
