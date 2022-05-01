import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_blobs

from gmm import GMM
from gmmReg import gmmReg, GMMParams

import copy

def main():
  # load means, covariances and weights
  mu_t = np.load('gmm_means.npy')
  print(f"mu_t shape: ", mu_t.shape)
  cov_t = np.load('gmm_covs.npy')
  cov_t = [cov_t[i][0][0] for i in range(cov_t.shape[0])]
  cov_t = np.array(cov_t)
  phi_t = np.load('gmm_weights.npy')
  print(f"target means: {mu_t}\n")
  print(f"target covs: {cov_t}\n")
  print(f"target weights: {phi_t}\n")

  # generate TF matrix
  r = 0
  p = 0
  y = 5
  r_euler = get_rot_euler(r, p, y)
  print("euler matrix from euler angles: \n", r_euler)
  q_r_euler = tf3d.quaternions.mat2quat(r_euler)
  print("quaternion vector: \n", q_r_euler)
  print()

  # transform target into source
  t = np.array([1.5, 1.5, 1.5])
  t = np.expand_dims(t, axis=1).T
  mu_s = mu_t + t
  cov_s = cov_t
  phi_s = phi_t
  print(f"source means: {mu_s}\n")
  print(f"source covs: {cov_s}\n")
  print(f"source weights: {phi_s}")

  # create GMMParam objects for each GMM
  tGMM = GMMParams(mu_t, cov_t, phi_t, mu_t.shape[0])
  sGMM = GMMParams(mu_s, cov_s, phi_s, mu_s.shape[0])

  # initial guess TF matrix
  qt = tf3d.quaternions.mat2quat(np.identity(3))
  # print(qt.shape, qt)
  qt = np.expand_dims(np.asarray(qt), axis=1)
  t_init = 0.25 * np.ones((3,1))
  tf_init = np.vstack((t_init, qt))
  # print(tf_init.shape, tf_init)

  # # begin L2 minimization procedure
  # cf = CostFunction()
  # gr = gmmReg(cf, tGMM, sGMM)
  # gr.register(tf_init)


if __name__ == "__main__":
  main()
