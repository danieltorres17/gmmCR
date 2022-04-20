import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

from gmm import GMM

def main():
  # number of clusters
  num_clusters = 3
  # get data
  X, y = make_blobs(n_samples=60, centers=num_clusters, n_features=3)
  # print("X shape: ", X.shape)
  # print("X: ")
  # print(X)
  # print()
  
  # get GMM object and fit on data
  gmm = GMM(X, num_clusters)
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
  gm_test = GaussianMixture(n_components=num_clusters).fit(X)
  print("sklearn GMM estimated means: ")
  print(gm_test.means_)
  print()
  print("sklearn GMM estimated covariances: ")
  print(gm_test.covariances_)
  print()
  print("sklearn GMM estimated weights: ")
  print(gm_test.weights_)

  # plot test data
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(X[:,0],X[:,1],X[:,2])
  # plt.grid()
  for l in range(len(gmm.cluster_means)):
    ax.scatter(gmm.cluster_means[l][0], gmm.cluster_means[l][1], gmm.cluster_means[l][2])
    ax.scatter(gm_test.means_[l][0], gm_test.means_[l][1], gm_test.means_[l][2], marker='x')
  plt.show()

if __name__ == "__main__":
  main()
