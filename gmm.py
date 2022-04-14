# %matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from sklearn.datasets import make_blobs
from sklearn.mixture import GaussianMixture

class GMM:
  def __init__(self, X, num_clusters, num_iters=100):
    # define number of clusters
    self.num_clusters = num_clusters
    # define number of samples
    self.num_samples = X.shape[0]
    # define number of iterations for EM
    self.num_iters = num_iters
    # initialize weights array with uniform probability
    self.weights = np.full((num_clusters,),fill_value=(1/num_clusters))
    # declare mean array
    self.cluster_means = []
    # declare covariance matrices array
    self.cluster_covs = []
    # declare cluster probability array
    self.cluster_probs = np.zeros((self.num_samples, self.num_clusters))
    # get means and covariances
    self.getMeansAndCovs(X)
    # declare variable to create track of convergence
    self.prev_weights_norm = None
    # convergence criteria 
    self.tol = 1e-8

  def getMeansAndCovs(self, X):
    # get random point indices from data
    rnd_idxs = np.random.choice(self.num_samples, size=self.num_clusters, replace=False)
    # get random points from data as initial cluster means
    self.cluster_means = X[rnd_idxs]
    # get covariance for each cluster
    self.cluster_covs = [np.cov(X.T) + np.eye(self.num_clusters) 
                         for _ in range(self.num_clusters)]

  def fit(self, X):
    for i in range(self.num_iters):
      # expectation step - calculate posterior probabilities
      self.expectation(X)
      # maximization step - relearn parameters
      self.maximization(X)
      # check for convergence
      if self.prev_weights_norm is None:
        self.prev_weights_norm = np.linalg.norm(self.weights)
      else:
        if abs(self.prev_weights_norm - np.linalg.norm(self.weights)) < self.tol:
          print(f"Converged at iteration: {i}")
          break

  def expectation(self, X):
    # array to store likelihoods 
    likelihoods = np.zeros((self.num_samples, self.num_clusters))
    # find likelihoods for each point in each cluster
    for k in range(self.num_clusters):
      likelihoods[:,k] = mvn.pdf(X, self.cluster_means[k], self.cluster_covs[k])
    # compute cluster posteriors
    self.cluster_probs = self.weights * likelihoods
    # normalize
    self.cluster_probs /= np.sum(self.cluster_probs, axis=1, keepdims=True)

  def maximization(self, X):
    # update cluster means
    for k in range(self.num_clusters):
      cp = np.expand_dims(self.cluster_probs[:,k], axis=1)
      self.cluster_means[k] = np.sum(cp * X, axis=0)
      self.cluster_means[k] /= np.sum(cp)

    # update cluster covariances
    for k in range(self.num_clusters):
      cp = np.expand_dims(self.cluster_probs[:,k], axis=1)
      cluster_cov = np.zeros((self.num_clusters, self.num_clusters))
      diff = X - self.cluster_means[k]
      cluster_cov = np.dot(diff.T, (diff*cp)) / np.sum(cp)
      self.cluster_covs[k] = cluster_cov

    # update cluster weights
    self.weights = np.mean(self.cluster_probs, axis=0)

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
