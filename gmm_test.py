import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from gmm import GMM

def main():
  # number of clusters
  num_clusters = 3
  # get data
  X, y = make_blobs(n_samples=60, centers=num_clusters, n_features=3, cluster_std=0.5)
  print("X shape: ", X.shape)
  # print("X: ")
  # print(X)
  # print()

  # # number of clusters to try
  # num_clusters_range = np.arange(start=2, stop=10)
  # # to store silhouette scores
  # sil_scores = []
  # # for each number of clusters, fit a KMeans classifier
  # for nc in num_clusters_range:
  #   # fit KMeans model
  #   model = KMeans(n_clusters=nc, random_state=0).fit(X)
  #   # make predictions on training data
  #   cluster_labels = model.fit_predict(X)
  #   # calculate silhouette score
  #   silhouette_avg = silhouette_score(X, cluster_labels)
  #   # store score
  #   sil_scores.append(silhouette_avg)

  # # find optimal number of clusters (arg of max silhouette score)
  # opt_num_clusters = np.argmax(sil_scores) + 2
  # # fit KMeans classifier with optimal number of clusters
  # model = KMeans(n_clusters=opt_num_clusters).fit(X)
  # print(f"Silhouette scores: {sil_scores}")
  # print(f"Optimal number of clusters: {opt_num_clusters}")
  # print("Cluster centers: ", model.cluster_centers_)
  
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
    ax.scatter(gmm.cluster_means[l][0], gmm.cluster_means[l][1], gmm.cluster_means[l][2], s=75)
    ax.scatter(gm_test.means_[l][0], gm_test.means_[l][1], gm_test.means_[l][2], marker='x', s=75)
  
  # set labels
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  plt.show()

if __name__ == "__main__":
  main()
