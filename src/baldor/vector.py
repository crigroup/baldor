#!/usr/bin/env python
"""
Vector operations
"""
import math
import numpy as np


def unit(vector):
  """
  Return vector divided by Euclidean (L2) norm

  Parameters
  ----------
  vector: array_like
    The input vector

  Returns
  -------
  unit : array_like
     vector divided by L2 norm

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> v0 = np.random.random(3)
  >>> v1 = br.vector.unit(v0)
  >>> np.allclose(v1, v0 / np.linalg.norm(v0))
  True
  """
  v = np.asarray(vector).squeeze()
  return v / math.sqrt((v**2).sum())

def norm(vector):
  """
  Return vector Euclidaan (L2) norm

  Parameters
  ----------
  vector: array_like
    The input vector

  Returns
  -------
  norm: float
   The computed norm

  Examples
  --------
  >>> import numpy as np
  >>> import baldor as br
  >>> v = np.random.random(3)
  >>> n = br.vector.norm(v)
  >>> numpy.allclose(n, np.linalg.norm(v))
  True
  """
  return math.sqrt((np.asarray(vector)**2).sum())

def perpendicular(vector):
  """
  Find an arbitrary perpendicular vector

  Parameters
  ----------
  vector: array_like
    The input vector

  Returns
  -------
  result: array_like
    The perpendicular vector
  """
  u = unit(vector)
  if np.allclose(u[:2], np.zeros(2)):
    if np.isclose(u[2], 0.):
      # unit is (0, 0, 0)
      raise ValueError('Input vector cannot be a zero vector')
    # unit is (0, 0, Z)
    result = np.array(br.Y_AXIS, dtype=np.float64, copy=True)
  result = np.array([-unit[1], unit[0], 0], dtype=np.float64)
  return result

def transform_between_vectors(vector_a, vector_b):
  """
  Compute the transformation that aligns two vectors

  Parameters
  ----------
  vector_a: array_like
    The initial vector
  vector_b: array_like
    The goal vector

  Returns
  -------
  transform: array_like
    The transformation between `vector_a` a `vector_b`
  """
  ua = unit(vector_a)
  ub = unit(vector_b)
  c = np.dot(ua, ub)
  angle = np.arccos(c)
  if np.isclose(c, -1.0) or np.allclose(ua, ub):
    axis = perpendicular(ub)
  else:
    axis = unit(np.cross(ua, ub))
  transform = br.axis_angle.to_transform(axis, angle)
  return transform
