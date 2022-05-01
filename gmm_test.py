import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle

from gmm import GMM

def main():
  # number of clusters
  num_clusters = 4
  # get data
  X, y = make_blobs(n_samples=60, centers=num_clusters, n_features=3, cluster_std=0.5)
  print("X shape: ", X.shape)
  # print("X: ")
  # print(X)
  # print()

  # get GMM object and fit on data
  gmm = GMM(X)
  # print(gmm.cluster_means)
  # print(len(gmm.cluster_covs))
  # print(gmm.weights)
  # print()

  # gmm.expectation(X)
  # print("gmm.cluster_probs: ")
  # print(gmm.cluster_probs)
  # print()
  # gmm.maximization(X)
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

  # comparison model
  gm_test = GaussianMixture(n_components=num_clusters, covariance_type='spherical').fit(X)
  # gm_test = GaussianMixture(n_components=num_clusters).fit(X)
  print("sklearn GMM estimated means: ")
  print(gm_test.means_)
  print()
  print("sklearn GMM estimated covariances: ")
  print(gm_test.covariances_)
  print()
  print("sklearn GMM estimated weights: ")
  print(gm_test.weights_)

  # save generated GMM parameters to file
  np.save('gmm_means.npy', gmm.cluster_means)
  np.save('gmm_covs.npy', gmm.cluster_covs)
  np.save('gmm_weights.npy', gmm.weights)

  # plot test data
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(X[:,0],X[:,1],X[:,2])
  # plt.grid()
  for l in range(len(gmm.cluster_means)):
    ax.scatter(gmm.cluster_means[l][0], gmm.cluster_means[l][1], gmm.cluster_means[l][2], s=75)
    ax.scatter(gm_test.means_[l][0], gm_test.means_[l][1], gm_test.means_[l][2], marker='x', s=75)
  
  # set labels
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  plt.show()

if __name__ == "__main__":
  main()
