import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tf_tools import *

def main():
  # load means, covariances and weights
  mu_t = np.load('gmm_multiple_sphere_means.npy')
  print(f"mu_t shape: ", mu_t.shape)
  cov_t = np.load('gmm_multiple_sphere_covs.npy')
  cov_t = [cov_t[i][0][0] for i in range(cov_t.shape[0])]
  cov_t = np.array(cov_t)
  phi_t = np.load('gmm_multiple_sphere_weights.npy')
  # print(f"target means shape: {mu_t.shape}\n")
  # print(f"target means: {mu_t}\n")
  # print(f"target covs: {cov_t}\n")
  # print(f"target weights: {phi_t}\n")

  # generate TF matrix
  r = 0
  p = 0
  y = 1
  r_euler = get_rot_euler(r, p, y)
  # print("euler matrix from euler angles: \n", r_euler)
  q_r_euler = tf3d.quaternions.mat2quat(r_euler)
  # print("quaternion vector: \n", q_r_euler)
  # print()

  # initial guess TF matrix
  # qt = tf3d.quaternions.mat2quat(np.identity(3))
  # print(qt.shape, qt)
  # qt = np.expand_dims(np.asarray(qt), axis=1)
  qt = get_quat_vector_from_mat(r_euler)
  qt = np.expand_dims(np.asarray(qt), axis=1)
  # print("qt: ", qt)
  t_init = 0.15 * np.ones((3,1))
  tf_init = np.vstack((qt, t_init))
  print("tf_init: \n", tf_init.T)

  # create initial guess with random noise
  t_guess = 0.1 * np.ones((3,1))
  guess_euler = get_rot_euler(0, 0, 0)
  qg = get_quat_vector_from_mat(guess_euler)
  qg = np.expand_dims(np.asarray(qg), axis=1)
  tf_guess = np.vstack((qg, t_guess))
  print("tf_guess: \n", tf_guess.T)
  print()

  # transform target into source
  t = t_init
  mu_s_t = transform(mu_t, r_euler, t)
  cov_s = cov_t
  phi_s = phi_t
  # print(f"source means: {mu_s_t}\n")
  # print(f"source covs: {cov_s}\n")
  # print(f"source weights: {phi_s}")

  print(mu_t.shape)
  print(mu_s_t.shape)

  # plot test data
  fig = plt.figure()
  ax = Axes3D(fig)
  # ax.scatter(X[:,0],X[:,1],X[:,2])

  for l in range(mu_t.shape[0]):
    ax.scatter(mu_t[l][0], mu_t[l][1], mu_t[l][2], s=300)
    ax.scatter(mu_s_t[l][0], mu_s_t[l][1], mu_s_t[l][2], marker='x', s=300)
  
  # set labels
  # ax.set_title("Num Clusters: 5")
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  ax.axis('tight')

  # plt.savefig("sphere_edges_gmm_test.png")
  plt.show()

  # # create GMMParam objects for each GMM
  # tGMM = GMMParams(mu_t, cov_t, phi_t, mu_t.shape[0])
  # sGMM = GMMParams(mu_s_t, cov_s, phi_s, mu_s_t.shape[0])

  # # begin L2 minimization procedure
  # print("Running solver!\n")
  # cf = CostFunction()
  # gr = gmmReg(cf, tGMM, sGMM)
  # solver, x_sol = gr.register(tf_guess)
  # print(f"Solver information: \n{solver}")
  # print(f"Solution found: \n{x_sol}")

  # # convert quaternions to euler angles
  # q_sol = x_sol[:4]
  # # print(q_sol)
  # rs, ps, ys = get_euler_from_quat(q_sol)
  # print(f"Roll found: {np.rad2deg(rs)}")
  # print(f"Pitch found: {np.rad2deg(ps)}")
  # print(f"Yaw found: {np.rad2deg(ys)}")

if __name__ == "__main__":
  main()
