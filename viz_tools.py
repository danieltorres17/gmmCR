import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_pts_and_means(X, gmm):
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(X[:,0],X[:,1],X[:,2])
  for l in range(len(gmm.cluster_means)):
    ax.scatter(gmm.cluster_means[l][0], gmm.cluster_means[l][1], gmm.cluster_means[l][2], s=75)
    # ax.scatter(gm_test.means_[l][0], gm_test.means_[l][1], gm_test.means_[l][2], marker='x', s=75)
  
  # set labels
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  plt.show()
