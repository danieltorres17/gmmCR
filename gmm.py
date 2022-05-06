# %matplotlib widget
import numpy as np
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

class GMM:
  def __init__(self, X, num_iters=100, cov_type="spherical"):
    # define number of features
    self.num_features = X.shape[1]
    # define covariance type
    self.cov_type = cov_type
    # declare mean array
    self.cluster_means = None
    # declare covariance matrices array
    self.cluster_covs = None
    # set cluster means, cluster covariances and set the number of clusters
    self.num_clusters = self.initMeansAndCovs(X)
    # define number of samples
    self.num_samples = X.shape[0]
    # define number of iterations for EM
    self.num_iters = num_iters
    # initialize weights array with uniform probability
    self.weights = np.full((self.num_clusters,),fill_value=(1/self.num_clusters))
    # declare cluster probability array
    self.cluster_probs = np.zeros((self.num_samples, self.num_clusters))
    # declare variable to create track of convergence
    self.prev_weights_norm = None
    # convergence criteria 
    self.tol = 1e-8

  def getMeansAndCovs(self, X):
    """
    Initialize the GMM means with randomly selected points from the data.
    Provides worse results than when using KMeans as initialization. 
    """
    # get random point indices from data
    rnd_idxs = np.random.choice(self.num_samples, size=self.num_clusters, replace=False)
    # get random points from data as initial cluster means
    self.cluster_means = X[rnd_idxs]
    # get covariance for each cluster
    self.cluster_covs = [np.cov(X.T) + np.eye(self.num_features) 
                         for _ in range(self.num_clusters)]

  def initMeansAndCovs(self, X, verbose=True):
    """
    Initialize the GMM means with the results from the KMeans classifier. 
    The optimal number of clusters for KMeans is found using the silhouette
    score. The code was adapted from:
    https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html 
    """
    # number of clusters to try
    # num_clusters_range = np.arange(start=10, stop=100, step=10)
    num_clusters_range = np.arange(start=2, stop=10, step=1)
    # to store silhouette scores
    sil_scores = []
    # for each number of clusters, fit a KMeans classifier
    for nc in num_clusters_range:
      # fit KMeans model
      model = KMeans(n_clusters=nc, random_state=0).fit(X)
      # make predictions on training data
      cluster_labels = model.fit_predict(X)
      # calculate silhouette score
      silhouette_avg = silhouette_score(X, cluster_labels)
      # store score
      sil_scores.append(silhouette_avg)

    # find optimal number of clusters (arg max of silhouette scores)
    opt_num_clusters = np.argmax(sil_scores) + 2
    # fit KMeans classifier with optimal number of clusters
    model = KMeans(n_clusters=opt_num_clusters).fit(X)
    # for debugging purposes
    if verbose:
      print(f"Silhouette scores: {sil_scores}")
      print(f"Optimal number of clusters: {opt_num_clusters}")
      print("Cluster centers: ", model.cluster_centers_)

    # set the GMM cluster means to be the KMeans cluster centers
    self.cluster_means = model.cluster_centers_
    # initialize covariances
    self.cluster_covs = [np.cov(X.T) + np.eye(self.num_features) 
                         for _ in range(opt_num_clusters)]

    return opt_num_clusters
    
  def fit(self, X):
    """
    Fit the GMM for a given number of iterations. 
    """
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
    # if spherical covariances (default)
    if self.cov_type == 'spherical':
      for k in range(self.num_clusters):
        cp = np.expand_dims(self.cluster_probs[:,k], axis=1)
        # print(f"cp dimensions: {cp.shape}")
        diff = np.linalg.norm(X - self.cluster_means[k], axis=1, keepdims=True)
        # print(f"diff dimensions: {diff.shape}")
        cluster_cov = np.dot(cp.T, diff) / (self.num_features * np.sum(cp))
        self.cluster_covs[k] = cluster_cov * np.eye(self.num_features)
    else:
      # full covariances
      for k in range(self.num_clusters):
        cp = np.expand_dims(self.cluster_probs[:,k], axis=1)
        cluster_cov = np.zeros((self.num_clusters, self.num_clusters))
        diff = X - self.cluster_means[k]
        cluster_cov = np.dot(diff.T, (diff*cp)) / np.sum(cp)
        self.cluster_covs[k] = cluster_cov

    # update cluster weights
    self.weights = np.mean(self.cluster_probs, axis=0)
