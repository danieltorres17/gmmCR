import numpy as np
from scipy.optimize import minimize
import time
from dataclasses import dataclass

from gmm import GMM

class gmmReg:
  """
  This class takes in two Gaussian Mixture Models as
  representations of the target and source point sets
  to be aligned 
  """
  def __init__(target_GMM, source_GMM, optim_iters=100, tol=1e-8):
    self.num_clusters = target_GMM.num_clusters
    self.target_GMM = target_GMM
    self.source_GMM = source_GMM
    self.optim_iters = optim_iters
    self.tol = tol

  def transform(self):
    pass

  def gauss_transform(self, source, target, weights, h):
    """
    source: source points
    target target points
    h: bandwidth parameter
    """
    pass

  def cost_function(self):
    pass

  def L2_dist(mu_s, phi_s, mu_t, phi_t, sigma):
    # compute normalizing constant
    Z = (2 * np.pi ** sigma**2) ** (0.5 * self.num_clusters/2)
    # compute Gauss transform
    gt = self.gauss_transform(mu_t, np.sqrt(2.0) * sigma)
    

  def register(self):
    pass
