#!/usr/bin/env python
"""
Axis-angle rotations
"""
import math
import numpy as np
# Local modules
import baldor as br


def to_euler(axis, angle, axes='sxyz'):
  """
  Return Euler angles from a rotation in the axis-angle representation.

  Parameters
  ----------
  axis: array_like
    axis around which the rotation occurs
  angle: float
    angle of rotation
  axes: str, optional
    Axis specification; one of 24 axis sequences as string or encoded tuple

  Returns
  -------
  ai: float
    First rotation angle (according to axes).
  aj: float
    Second rotation angle (according to axes).
  ak: float
    Third rotation angle (according to axes).
  """
  T = br.axis_angle.to_transform(axis, angle)
  return br.transform.to_euler(T, axes)

def to_quaternion(axis, angle, isunit=False):
  """
  Return quaternion from a rotation in the axis-angle representation.

  Parameters
  ----------
  axis: array_like
    axis around which the rotation occurs
  angle: float
    angle of rotation

  Returns
  -------
  q: array_like
    Quaternion in w, x, y z (real, then vector) format

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
  """
  u = np.array(axis)
  if not isunit:
    # Cannot divide in-place because input vector may be integer type,
    # whereas output will be float type; this may raise an error in versions
    # of numpy > 1.6.1
    u = u / math.sqrt(np.dot(u, u))
  t2 = angle / 2.
  st2 = math.sin(t2)
  return np.concatenate(([math.cos(t2)], u*st2))

def to_transform(axis, angle, point=None):
  """
  Return homogeneous transformation from an axis-angle rotation.

  Parameters
  ----------
  axis: array_like
    axis around which the rotation occurs
  angle: float
    angle of rotation
  point: array_like
    point around which the rotation is performed

  Returns
  -------
  T: array_like
    Homogeneous transformation (4x4)
  """
  sina = math.sin(angle)
  cosa = math.cos(angle)
  axis = br.vector.unit(axis[:3])
  # rotation matrix around unit vector
  R = np.diag([cosa, cosa, cosa])
  R += np.outer(axis, axis) * (1.0 - cosa)
  axis *= sina
  R += np.array([ [ 0.0,    -axis[2], axis[1]],
                  [ axis[2], 0.0,     -axis[0]],
                  [-axis[1], axis[0],  0.0]])
  T = np.identity(4)
  T[:3, :3] = R
  if point is not None:
    # rotation not around origin
    point = np.array(point[:3], dtype=np.float64, copy=False)
    T[:3, 3] = point - np.dot(R, point)
  return T
