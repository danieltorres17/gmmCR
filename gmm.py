# %matplotlib widget
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn

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
