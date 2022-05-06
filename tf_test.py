import numpy as np
from scipy.spatial.transform import Rotation as R

def transform(data, t, roll, pitch, yaw):
  r = R.from_euler('zyx', [roll, pitch, yaw], degrees=True)
  
  rot = r.as_matrix()

  t = np.expand_dims(t, axis=1)
  tf_data = np.dot(rot, data) + t

  return tf_data.T

if __name__ == "__main__":
  d = np.array([[1, 1, 1], [1,1,1]])
  print(d.shape)
  print(d)
  t = np.array([1,1,2])
  print(t.shape)
  tf_d = transform(d.T, t, 90, 0, 0)
  print("d: ", d)
  print(tf_d.shape)
  print("tf_d: ", tf_d)
