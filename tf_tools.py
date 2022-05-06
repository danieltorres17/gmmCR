from scipy.spatial.transform import Rotation as R
import transforms3d as tf3d
import numpy as np

def get_rot_mat(q):
  """
  Return rotation matrix from quaternions
  """
  rot = tf3d.quaternions.quat2mat(q)
  rot = np.asarray(rot)
  return rot

def get_euler_from_quat(q):
  """
  Convert quaternion vector to euler angles
  """
  angles = tf3d.euler.quat2euler(q)
  return angles[0], angles[1], angles[2]

def transform(data, R, t):
  """
  Transform set of points
  """
  # apply rotation and translation to all vectors 
  if t.ndim == 1:
    t = np.expand_dims(t, axis=1)
  tf_data = np.dot(R, data.T) + t

  return tf_data.T

def get_quat_vector_from_mat(R):
  """
  Get quaternion vector from rotation matrix
  """
  q = tf3d.quaternions.mat2quat(R)
  return q

def get_rot_euler(roll, pitch, yaw):
  """
  Get rotation matrix from euler angles
  """
  r = R.from_euler('xyz', np.array([roll, pitch, yaw]), degrees=True)
  return np.asarray(r.as_matrix(), dtype=np.float64)

def diff_rot_from_quaternion(q):
  """Differential rotation matrix from quaternion.
  dR(q)/dq = [dR(q)/dq0, dR(q)/dq1, dR(q)/dq2, dR(q)/dq3]
  Args:
  q (numpy.ndarray): Quaternion.
  Modified from:
  https://github.com/neka-nat/probreg/blob/master/probreg/se3_op.py
  """
  rot = tf3d.quaternions.quat2mat(q)
  q2 = np.square(q)
  z = np.sum(q2)
  z2 = z * z
  d_rot = np.zeros((4, 3, 3))
  d_rot[0, 0, 0] = 4 * q[0] * (q2[2] + q2[3]) / z2
  d_rot[1, 0, 0] = 4 * q[1] * (q2[2] + q2[3]) / z2
  d_rot[2, 0, 0] = -4 * q[2] * (q2[1] + q2[0]) / z2
  d_rot[3, 0, 0] = -4 * q[3] * (q2[1] + q2[0]) / z2

  d_rot[0, 1, 1] = 4 * q[0] * (q2[1] + q2[3]) / z2
  d_rot[1, 1, 1] = -4 * q[1] * (q2[2] + q2[0]) / z2
  d_rot[2, 1, 1] = 4 * q[2] * (q2[1] + q2[3]) / z2
  d_rot[3, 1, 1] = -4 * q[3] * (q2[2] + q2[0]) / z2

  d_rot[0, 2, 2] = 4 * q[0] * (q2[1] + q2[2]) / z2
  d_rot[1, 2, 2] = -4 * q[1] * (q2[3] + q2[0]) / z2
  d_rot[2, 2, 2] = -4 * q[2] * (q2[1] + q2[2]) / z2
  d_rot[3, 2, 2] = 4 * q[3] * (q2[3] + q2[0]) / z2

  d_rot[0, 0, 1] = -2 * q[3] / z - 2 * q[0] * rot[0, 1] / z2
  d_rot[1, 0, 1] = 2 * q[2] / z - 2 * q[1] * rot[0, 1] / z2
  d_rot[2, 0, 1] = 2 * q[1] / z - 2 * q[2] * rot[0, 1] / z2
  d_rot[3, 0, 1] = -2 * q[0] / z - 2 * q[3] * rot[0, 1] / z2

  d_rot[0, 0, 2] = 2 * q[2] / z - 2 * q[0] * rot[0, 2] / z2
  d_rot[1, 0, 2] = 2 * q[3] / z - 2 * q[1] * rot[0, 2] / z2
  d_rot[2, 0, 2] = 2 * q[0] / z - 2 * q[2] * rot[0, 2] / z2
  d_rot[3, 0, 2] = 2 * q[1] / z - 2 * q[3] * rot[0, 2] / z2

  d_rot[0, 1, 0] = 2 * q[3] / z - 2 * q[0] * rot[1, 0] / z2
  d_rot[1, 1, 0] = 2 * q[2] / z - 2 * q[1] * rot[1, 0] / z2
  d_rot[2, 1, 0] = 2 * q[1] / z - 2 * q[2] * rot[1, 0] / z2
  d_rot[3, 1, 0] = 2 * q[0] / z - 2 * q[3] * rot[1, 0] / z2

  d_rot[0, 1, 2] = -2 * q[1] / z - 2 * q[0] * rot[1, 2] / z2
  d_rot[1, 1, 2] = -2 * q[0] / z - 2 * q[1] * rot[1, 2] / z2
  d_rot[2, 1, 2] = 2 * q[3] / z - 2 * q[2] * rot[1, 2] / z2
  d_rot[3, 1, 2] = 2 * q[2] / z - 2 * q[3] * rot[1, 2] / z2

  d_rot[0, 2, 0] = -2 * q[2] / z - 2 * q[0] * rot[2, 0] / z2
  d_rot[1, 2, 0] = 2 * q[3] / z - 2 * q[1] * rot[2, 0] / z2
  d_rot[2, 2, 0] = -2 * q[0] / z - 2 * q[2] * rot[2, 0] / z2
  d_rot[3, 2, 0] = 2 * q[1] / z - 2 * q[3] * rot[2, 0] / z2

  d_rot[0, 2, 1] = 2 * q[1] / z - 2 * q[0] * rot[2, 1] / z2
  d_rot[1, 2, 1] = 2 * q[0] / z - 2 * q[1] * rot[2, 1] / z2
  d_rot[2, 2, 1] = 2 * q[3] / z - 2 * q[2] * rot[2, 1] / z2
  d_rot[3, 2, 1] = 2 * q[2] / z - 2 * q[3] * rot[2, 1] / z2

  return d_rot

