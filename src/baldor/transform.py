#!/usr/bin/env python
"""
Homogeneous Transformation Matrices
"""
import math
import numpy as np
# Local modules
import baldor as br


def are_equal(T1, T2, rtol=1e-5, atol=1e-8):
  """
  Returns True if two homogeneous transformation are equal within a tolerance.

  Parameters
  ----------
  T1: array_like
    First input homogeneous transformation
  T2: array_like
    Second input homogeneous transformation
  rtol: float
    The relative tolerance parameter.
  atol: float
    The absolute tolerance parameter.

  Returns
  -------
  equal : bool
     True if `T1` and `T2` are `almost` equal, False otherwise

  See Also
  --------
  numpy.allclose: Contains the details about the tolerance parameters
  """
  M1 = np.array(T1, dtype=np.float64, copy=True)
  M1 /= M1[3,3]
  M2 = np.array(T2, dtype=np.float64, copy=True)
  M2 /= M2[3,3]
  return np.allclose(M1, M2, rtol, atol)

def between_axes(axis_a, axis_b):
  """
  Compute the transformation that aligns two vectors/axes.

  Parameters
  ----------
  axis_a: array_like
    The initial axis
  axis_b: array_like
    The goal axis

  Returns
  -------
  transform: array_like
    The transformation that transforms `axis_a` into `axis_b`
  """
  a_unit = br.vector.unit(axis_a)
  b_unit = br.vector.unit(axis_b)
  c = np.dot(a_unit, b_unit)
  angle = np.arccos(c)
  if np.isclose(c, -1.0) or np.allclose(a_unit, b_unit):
    axis = br.vector.perpendicular(b_unit)
  else:
    axis = br.vector.unit(np.cross(a_unit, b_unit))
  transform = br.axis_angle.to_transform(axis, angle)
  return transform

def inverse(transform):
  """
  Compute the inverse of an homogeneous transformation.

  .. note:: This function is more efficient than :obj:`numpy.linalg.inv` given
    the special properties of homogeneous transformations.

  Parameters
  ----------
  transform: array_like
    The input homogeneous transformation

  Returns
  -------
  inv: array_like
    The inverse of the input homogeneous transformation
  """
  R = transform[:3,:3].T
  p = transform[:3,3]
  inv = np.eye(4)
  inv[:3,:3] = R
  inv[:3,3] = np.dot(-R, p)
  return inv

def random(max_position=1.):
  """
  Generate a random homogeneous transformation.

  Parameters
  ----------
  max_position: float, optional
    Maximum value for the position components of the transformation

  Returns
  -------
  T: array_like
    The random homogeneous transformation

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> T = br.transform.random()
  >>> Tinv = br.transform.inverse(T)
  >>> np.allclose(np.dot(T, Tinv), np.eye(4))
  True
  """
  quat = br.quaternion.random()
  T = br.quaternion.to_transform(quat)
  T[:3,3] = np.random.rand(3)*max_position
  return T


def to_axis_angle(transform):
  """
  Return rotation angle and axis from rotation matrix.

  Parameters
  ----------
  transform: array_like
    The input homogeneous transformation

  Returns
  -------
  axis: array_like
    axis around which rotation occurs
  angle: float
    angle of rotation
  point: array_like
    point around which the rotation is performed

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> axis = np.random.sample(3) - 0.5
  >>> angle = (np.random.sample(1) - 0.5) * (2*np.pi)
  >>> point = np.random.sample(3) - 0.5
  >>> T0 = br.axis_angle.to_transform(axis, angle, point)
  >>> axis, angle, point = br.transform.to_axis_angle(T0)
  >>> T1 = br.axis_angle.to_transform(axis, angle, point)
  >>> br.transform.are_equal(T0, T1)
  True
  """
  R = np.array(transform, dtype=np.float64, copy=False)
  R33 = R[:3,:3]
  # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
  w, W = np.linalg.eig(R33.T)
  i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
  axis = np.real(W[:, i[-1]]).squeeze()
  # point: unit eigenvector of R corresponding to eigenvalue of 1
  w, Q = np.linalg.eig(R)
  i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
  point = np.real(Q[:, i[-1]]).squeeze()
  point /= point[3]
  # rotation angle depending on axis
  cosa = (np.trace(R33) - 1.0) / 2.0
  if abs(axis[2]) > 1e-8:
    sina = (R[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
  elif abs(axis[1]) > 1e-8:
    sina = (R[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
  else:
    sina = (R[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]
  angle = math.atan2(sina, cosa)
  return axis, angle, point

def to_dual_quaternion(transform):
  """
  Return quaternion from the rotation part of an homogeneous transformation.

  Parameters
  ----------
  transform: array_like
    Rotation matrix. It can be (3x3) or (4x4)
  isprecise: bool
    If True, the input transform is assumed to be a precise rotation matrix and
    a faster algorithm is used.

  Returns
  -------
  qr: array_like
    Quaternion in w, x, y z (real, then vector) for the rotation component
  qt: array_like
    Quaternion in w, x, y z (real, then vector) for the translation component

  Notes
  -----
  Some literature prefers to use :math:`q` for the rotation component and
  :math:`q'` for the translation component
  """
  cot = lambda x: 1./np.tan(x)
  R = np.eye(4)
  R[:3,:3] = transform[:3,:3]
  l,theta,_ = to_axis_angle(R)
  t = transform[:3,3]
  # Pitch d
  d = np.dot(l.reshape(1,3), t.reshape(3,1))
  # Point c
  c = 0.5*(t-d*l) + cot(theta/2.)*np.cross(l,t)
  # Moment vector
  m = np.cross(c, l)
  # Rotation quaternion
  qr = np.zeros(4)
  qr[0] = np.cos(theta/2.)
  qr[1:] = np.sin(theta/2.)*l
  # Translation quaternion
  qt = np.zeros(4)
  qt[0] = -(1/2.)*np.dot(qr[1:],t)
  qt[1:] = (1/2.)*(qr[0]*t + np.cross(t,qr[1:]))
  return qr, qt

def to_euler(transform, axes='sxyz'):
  """
  Return Euler angles from transformation matrix with the specified axis
  sequence.

  Parameters
  ----------
  transform: array_like
    Rotation matrix. It can be (3x3) or (4x4)
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

  Notes
  -----
  Many Euler angle triplets can describe the same rotation matrix

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> T0 = br.euler.to_transform(1, 2, 3, 'syxz')
  >>> al, be, ga = br.transform.to_euler(T0, 'syxz')
  >>> T1 = br.euler.to_transform(al, be, ga, 'syxz')
  >>> np.allclose(T0, T1)
  True
  """
  try:
    firstaxis, parity, repetition, frame = br._AXES2TUPLE[axes.lower()]
  except (AttributeError, KeyError):
    br._TUPLE2AXES[axes]  # validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis
  j = br._NEXT_AXIS[i+parity]
  k = br._NEXT_AXIS[i-parity+1]

  M = np.array(transform, dtype=np.float64, copy=False)[:3,:3]
  if repetition:
    sy = math.sqrt(M[i,j]*M[i,j] + M[i,k]*M[i,k])
    if sy > br._EPS:
      ax = math.atan2( M[i,j],  M[i,k])
      ay = math.atan2( sy,      M[i,i])
      az = math.atan2( M[j,i], -M[k,i])
    else:
      ax = math.atan2(-M[j,k],  M[j,j])
      ay = math.atan2( sy,      M[i,i])
      az = 0.0
  else:
    cy = math.sqrt(M[i,i]*M[i,i] + M[j,i]*M[j,i])
    if cy > br._EPS:
      ax = math.atan2( M[k,j],  M[k,k])
      ay = math.atan2(-M[k,i],  cy)
      az = math.atan2( M[j,i],  M[i,i])
    else:
      ax = math.atan2(-M[j,k],  M[j,j])
      ay = math.atan2(-M[k,i],  cy)
      az = 0.0

  if parity:
    ax, ay, az = -ax, -ay, -az
  if frame:
    ax, az = az, ax
  return ax, ay, az

def to_quaternion(transform, isprecise=False):
  """
  Return quaternion from the rotation part of an homogeneous transformation.

  Parameters
  ----------
  transform: array_like
    Rotation matrix. It can be (3x3) or (4x4)
  isprecise: bool
    If True, the input transform is assumed to be a precise rotation matrix and
    a faster algorithm is used.

  Returns
  -------
  q: array_like
    Quaternion in w, x, y z (real, then vector) format

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> q = br.transform.to_quaternion(np.identity(4), isprecise=True)
  >>> np.allclose(q, [1, 0, 0, 0])
  True
  >>> q = br.transform.to_quaternion(np.diag([1, -1, -1, 1]))
  >>> np.allclose(q, [0, 1, 0, 0]) or np.allclose(q, [0, -1, 0, 0])
  True
  >>> T = br.axis_angle.to_transform((1, 2, 3), 0.123)
  >>> q = br.transform.to_quaternion(T, True)
  >>> np.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
  True
  """
  M = np.array(transform, dtype=np.float64, copy=False)[:4, :4]
  if isprecise:
    q = np.empty((4, ))
    t = np.trace(M)
    if t > M[3, 3]:
      q[0] = t
      q[3] = M[1, 0] - M[0, 1]
      q[2] = M[0, 2] - M[2, 0]
      q[1] = M[2, 1] - M[1, 2]
    else:
      i, j, k = 0, 1, 2
      if M[1, 1] > M[0, 0]:
        i, j, k = 1, 2, 0
      if M[2, 2] > M[i, i]:
        i, j, k = 2, 0, 1
      t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
      q[i] = t
      q[j] = M[i, j] + M[j, i]
      q[k] = M[k, i] + M[i, k]
      q[3] = M[k, j] - M[j, k]
      q = q[[3, 0, 1, 2]]
    q *= 0.5 / math.sqrt(t * M[3, 3])
  else:
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                  [m01+m10,     m11-m00-m22, 0.0,         0.0],
                  [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                  [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
  if q[0] < 0.0:
    np.negative(q, q)
  return q
