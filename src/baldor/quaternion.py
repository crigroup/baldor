#!/usr/bin/env python
"""
Functions to operate quaternions.

.. important:: Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
"""
import math
import numpy as np
# Local modules
import baldor as br


def are_equal(q1, q2, rtol=1e-5, atol=1e-8):
  """
  Returns True if two quaternions are equal within a tolerance.

  Parameters
  ----------
  q1: array_like
    First input quaternion (4 element sequence)
  q2: array_like
    Second input quaternion (4 element sequence)
  rtol: float
    The relative tolerance parameter.
  atol: float
    The absolute tolerance parameter.

  Returns
  -------
  equal : bool
     True if `q1` and `q2` are `almost` equal, False otherwise

  See Also
  --------
  numpy.allclose: Contains the details about the tolerance parameters

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

  Examples
  --------
  >>> import baldor as br
  >>> q1 = [1, 0, 0, 0]
  >>> br.quaternion.are_equal(q1, [0, 1, 0, 0])
  False
  >>> br.quaternion.are_equal(q1, [1, 0, 0, 0])
  True
  >>> br.quaternion.are_equal(q1, [-1, 0, 0, 0])
  True
  """
  if np.allclose(q1, q2, rtol, atol):
    return True
  return np.allclose(np.array(q1)*-1, q2, rtol, atol)

def conjugate(q):
  """
  Compute the conjugate of a quaternion.

  Parameters
  ----------
  q: array_like
    Input quaternion (4 element sequence)

  Returns
  -------
  qconj: ndarray
    The conjugate of the input quaternion.

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

  Examples
  --------
  >>> import baldor as br
  >>> q0 = br.quaternion.random()
  >>> q1 = br.quaternion.conjugate(q0)
  >>> q1[0] == q0[0] and all(q1[1:] == -q0[1:])
  True

  """
  qconj = np.array(q, dtype=np.float64, copy=True)
  np.negative(qconj[1:], qconj[1:])
  return qconj

def dual_to_transform(qr, qt):
  """
  Return a homogeneous transformation from the given dual quaternion.

  Parameters
  ----------
  qr: array_like
    Input quaternion for the rotation component (4 element sequence)
  qt: array_like
    Input quaternion for the translation component (4 element sequence)

  Returns
  -------
  T: array_like
    Homogeneous transformation (4x4)

  Notes
  -----
  Some literature prefers to use :math:`q` for the rotation component and
  :math:`q'` for the translation component
  """
  T = np.eye(4)
  R = br.quaternion.to_transform(qr)[:3,:3]
  t = 2*br.quaternion.multiply(qt, br.quaternion.conjugate(qr))
  T[:3,:3] = R
  T[:3,3] = t[1:]
  return T

def inverse(q):
  """
  Return multiplicative inverse of a quaternion

  Parameters
  ----------
  q: array_like
    Input quaternion (4 element sequence)

  Returns
  -------
  qinv : ndarray
     The inverse of the input quaternion.

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
  """
  return conjugate(q) / norm(q)

def multiply(q1, q2):
  """
  Multiply two quaternions

  Parameters
  ----------
  q1: array_like
    First input quaternion (4 element sequence)
  q2: array_like
    Second input quaternion (4 element sequence)

  Returns
  -------
  result: ndarray
    The resulting quaternion

  Notes
  -----
  `Hamilton product of quaternions
  <http://en.wikipedia.org/wiki/Quaternions#Hamilton_product>`_

  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> q = br.quaternion.multiply([4, 1, -2, 3], [8, -5, 6, 7])
  >>> np.allclose(q, [28, -44, -14, 48])
  True
  """
  w1, x1, y1, z1 = q1
  w2, x2, y2, z2 = q2
  return np.array([ -x1*x2 - y1*y2 - z1*z2 + w1*w2,
                    x1*w2 + y1*z2 - z1*y2 + w1*x2,
                    -x1*z2 + y1*w2 + z1*x2 + w1*y2,
                    x1*y2 - y1*x2 + z1*w2 + w1*z2], dtype=np.float64)

def norm(q):
  """
  Compute quaternion norm

  Parameters
  ----------
  q : array_like
    Input quaternion (4 element sequence)

  Returns
  -------
  n : float
    quaternion norm

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
  """
  return np.dot(q, q)


def random(rand=None):
  """
  Generate an uniform random unit quaternion.

  Parameters
  ----------
  rand: array_like or None
    Three independent random variables that are uniformly distributed
    between 0 and 1.

  Returns
  -------
  qrand: array_like
    The random quaternion

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> q = br.quaternion.random()
  >>> np.allclose(1, np.linalg.norm(q))
  True
  """
  if rand is None:
    rand = np.random.rand(3)
  else:
    assert len(rand) == 3
  r1 = np.sqrt(1.0 - rand[0])
  r2 = np.sqrt(rand[0])
  pi2 = math.pi * 2.0
  t1 = pi2 * rand[1]
  t2 = pi2 * rand[2]
  return np.array([np.cos(t2)*r2, np.sin(t1)*r1, np.cos(t1)*r1, np.sin(t2)*r2])

def to_axis_angle(quaternion, identity_thresh=None):
  """
  Return axis-angle rotation from a quaternion

  Parameters
  ----------
  quaternion: array_like
    Input quaternion (4 element sequence)
  identity_thresh : None or scalar, optional
    Threshold below which the norm of the vector part of the quaternion (x,
       y, z) is deemed to be 0, leading to the identity rotation.  None (the
       default) leads to a threshold estimated based on the precision of the
       input.

  Returns
  ----------
  axis: array_like
    axis around which rotation occurs
  angle: float
    angle of rotation

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.
  A quaternion for which x, y, z are all equal to 0, is an identity rotation.
  In this case we return a `angle=0` and `axis=[1, 0, 0]``. This is an arbitrary
  vector.

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> axis, angle = br.euler.to_axis_angle(0, 1.5, 0, 'szyx')
  >>> np.allclose(axis, [0, 1, 0])
  True
  >>> angle
  1.5
  """
  w, x, y, z = quaternion
  Nq = norm(quaternion)
  if not np.isfinite(Nq):
    return np.array([1.0, 0, 0]), float('nan')
  if identity_thresh is None:
    try:
      identity_thresh = np.finfo(Nq.type).eps * 3
    except (AttributeError, ValueError): # Not a numpy type or not float
      identity_thresh = br._FLOAT_EPS * 3
  if Nq < br._FLOAT_EPS ** 2:  # Results unreliable after normalization
    return np.array([1.0, 0, 0]), 0.0
  if not np.isclose(Nq, 1):  # Normalize if not normalized
    s = math.sqrt(Nq)
    w, x, y, z = w / s, x / s, y / s, z / s
  len2 = x*x + y*y + z*z
  if len2 < identity_thresh**2:
      # if vec is nearly 0,0,0, this is an identity rotation
      return np.array([1.0, 0, 0]), 0.0
  # Make sure w is not slightly above 1 or below -1
  theta = 2 * math.acos(max(min(w, 1), -1))
  return  np.array([x, y, z]) / math.sqrt(len2), theta

def to_euler(quaternion, axes='sxyz'):
  """
  Return Euler angles from a quaternion using the specified axis sequence.

  Parameters
  ----------
  q : array_like
    Input quaternion (4 element sequence)
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
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> ai, aj, ak = br.quaternion.to_euler([0.99810947, 0.06146124, 0, 0])
  >>> np.allclose([ai, aj, ak], [0.123, 0, 0])
  True
  """
  return br.transform.to_euler(to_transform(quaternion), axes)

def to_transform(quaternion):
  """
  Return homogeneous transformation from a quaternion.

  Parameters
  ----------
  quaternion: array_like
    Input quaternion (4 element sequence)
  axes: str, optional
    Axis specification; one of 24 axis sequences as string or encoded tuple

  Returns
  -------
  T: array_like
    Homogeneous transformation (4x4)

  Notes
  -----
  Quaternions :math:`w + ix + jy + kz` are represented as :math:`[w, x, y, z]`.

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> T0 = br.quaternion.to_transform([1, 0, 0, 0]) # Identity quaternion
  >>> np.allclose(T0, np.eye(4))
  True
  >>> T1 = br.quaternion.to_transform([0, 1, 0, 0]) # 180 degree rot around X
  >>> np.allclose(T1, np.diag([1, -1, -1, 1]))
  True
  """
  q = np.array(quaternion, dtype=np.float64, copy=True)
  n = np.dot(q, q)
  if n < br._EPS:
    return np.identity(4)
  q *= math.sqrt(2.0 / n)
  q = np.outer(q, q)
  return np.array([
      [1.0-q[2,2]-q[3,3],     q[1,2]-q[3,0],      q[1,3]+q[2,0], 0.0],
      [    q[1,2]+q[3,0], 1.0-q[1,1]-q[3,3],      q[2,3]-q[1,0], 0.0],
      [    q[1,3]-q[2,0],     q[2,3]+q[1,0], 1.0-q[1,1]-q[2, 2], 0.0],
      [              0.0,               0.0,                0.0, 1.0]])
