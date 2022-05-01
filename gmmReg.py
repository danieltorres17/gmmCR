import numpy as np
from scipy.optimize import minimize

from gmm import GMM

class GMMParams:
  """
  To store GMM parameters
  """
  def __init__(self, means, covs, weights, num_clusters):
    self.means = means
    self.covs = covs
    self.weights = weights
    self.num_clusters = num_clusters

class gmmReg:
  """
  This class takes in two Gaussian Mixture Models as
  representations of the target and source point sets
  to be aligned 
  """
  def __init__(self, cost_fun, target_GMM, source_GMM, optim_iters=10):
    # set the cost function
    self.cost_fun = cost_fun
    # set the number of clusters
    self.num_clusters = target_GMM.num_clusters
    # set the target cluster parameters
    self.mu_t = target_GMM.means
    self.cov_t = target_GMM.covs
    self.phi_t = target_GMM.weights
    # set the source cluster parameters
    self.mu_s = source_GMM.means
    self.cov_s = source_GMM.covs
    self.phi_s = source_GMM.weights
    # optimizer parameters
    self.optim_iters = optim_iters
    self.tol = 1e-8
    self.sigma = 1.0

  def L2_dist(self, mu_s, phi_s, mu_t, phi_t, sigma):
    # transform source GMM cluster means
    R_est = np.eye(mu_s.shape[1])
    t_est = np.ones((3,))
    # mu_s_t = transform(mu_s, R_est, t_est)
    mu_s_t = mu_s
    # compute normalizing constant
    Z = (2 * np.pi * sigma**2) ** (0.5 * self.num_clusters/2)
    # define scale
    h = 1.0

    # compute Gauss transforms 
    f, Gf = self.gauss_transform(mu_t, mu_t, phi_t, phi_t, h, Z)
    print(f"f cost: {f}")
    fg, Gfg = self.gauss_transform(mu_t, mu_s_t, phi_t, phi_t, h, Z)
    print(f"fg cost: {fg}")
    g, Gg = self.gauss_transform(mu_s, mu_s, phi_s, phi_s, h, Z)
    print(f"g cost: {g}")
    # L2 distance cost between GMMs
    cost = f - 2*fg + g
    print(f"total cost: {cost}")

    # calculate gradients
    print("Gf dimensions: ", Gf.shape)
    print("mu_s dimensions: ", mu_s.shape)
    # df/dt
    dfdt = np.dot(Gf.T, np.ones((self.num_clusters, 1)))
    print("dfdt: ", dfdt)

    return cost

  def register(self, x_init):
    f = None
    # solve L2 minimization problem via L-BFGS-B solver
    for i in range(self.optim_iters):
      print("running solver")
      args = (self.mu_s, self.phi_s, self.mu_t, self.phi_t, self.sigma)
      res = minimize(self.cost_fun, x_init, args=args, 
                     method='BFGS', jac=True, tol=self.tol,
                     options={'maxiter': self.optim_iters})
      # annealing step
      self.sigma *= 0.9
      if not f is None and abs(res.fun - f) < self.tol:
        break
      f = res.fun
      x_init = res.x

    print("Solution found: ", res.x)
    print(res)

class CostFunction:
  def __init__(self):
    self.num_clusters = 4
  
  def gauss_transform(self, mu_t, mu_s, phi_t, phi_s, h, Z):
    """
    source: source points
    target target points
    h: squared sigma value
    """
    # calculate cost and gradient
    cost = 0.0 
    # to store G matrix
    G = np.zeros((self.num_clusters, mu_t.shape[1]))
    # calculate costs
    for i in range(self.num_clusters):
      for j in range(self.num_clusters):
        # difference between target and transformed source
        diff = np.subtract(mu_s[i], mu_t[j])
        # normalized GMM weights 
        w = (phi_t[i] * phi_s[j]) / Z
        # running cost
        cost += -w * np.exp(-np.sum(np.square(diff) / (2 * h**2)))

        # accumulate G
        G[i,:] -= cost * diff

    # apply normalization to G
    G *= (-1 / (2 * h**2))

    return cost, G

  def __call__(self, theta, *args):
    mu_t, phi_t, mu_s, phi_s, sigma = args
    # get translation vector
    t = theta[:3]
    # get rotation matrix
    rot = get_rot_mat(theta[:4])
    # transform source GMM
    mu_s_t = transform(mu_s, rot, t)
    # compute normalizing constant
    Z = (2 * np.pi * sigma**2) ** (0.5 * self.num_clusters/2)
    # define scale
    h = sigma
    # get L2 dist
    f, G = self.gauss_transform(mu_t, mu_s_t, phi_t, phi_s, h, Z)
    # dfdt = np.dot(G.T, np.ones((self.num_clusters, 1)))
    # get diff quaternion rotation matrix
    d_rot = diff_rot_from_quaternion(theta[:4])
    # calculate <G.T, M0>
    gtm0 = np.dot(G.T, mu_s)
    # calculate gradient
    grad = np.concatenate([(gtm0 * d_rot).sum(axis=(1, 2)), G.sum(axis=0)])
    
    return f, grad