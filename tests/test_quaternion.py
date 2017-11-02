#! /usr/bin/env python
import math
import unittest
import numpy as np
# Tested package
import baldor as br


class TestModule(unittest.TestCase):
  def test_are_equal(self):
    q1 = [1, 0, 0, 0]
    self.assertFalse(br.quaternion.are_equal(q1, [0, 1, 0, 0]))
    self.assertTrue(br.quaternion.are_equal(q1, [1, 0, 0, 0]))
    self.assertTrue(br.quaternion.are_equal(q1, [-1, 0, 0, 0]))

  def test_conjugate(self):
    q0 = br.quaternion.random()
    q1 = br.quaternion.conjugate(q0)
    np.testing.assert_allclose(q0[0], q1[0])
    np.testing.assert_allclose(-q0[1:], q1[1:])

  def test_inverse(self):
    q0 = br.quaternion.random()
    q1 = br.quaternion.inverse(q0)
    quat_mult = br.quaternion.multiply(q0, q1)
    np.testing.assert_allclose(quat_mult, [1, 0, 0, 0], rtol=1e-7, atol=1e-8)

  def test_multipy(self):
    q = br.quaternion.multiply([4, 1, -2, 3], [8, -5, 6, 7])
    np.testing.assert_allclose(q, [28, -44, -14, 48])

  def test_norm(self):
    for _ in range(100):
      q = br.quaternion.random()
      np.testing.assert_allclose(br.quaternion.norm(q), 1)

  def test_random(self):
    for _ in range(100):
      q = br.quaternion.random()
      np.testing.assert_allclose(np.linalg.norm(q), 1)
    # Test invalid seed lenght
    def invalid_rand_seed():
      br.quaternion.random(rand=[1,2])
    self.assertRaises(AssertionError, invalid_rand_seed)

  def test_to_axis_angle(self):
    axis, angle = br.quaternion.to_axis_angle([0, 1, 0, 0])
    np.testing.assert_allclose(axis, [1, 0, 0])
    np.testing.assert_almost_equal(angle, np.pi)
    # Identity
    axis, angle = br.quaternion.to_axis_angle([1, 0, 0, 0])
    np.testing.assert_allclose(axis, [1, 0, 0])
    np.testing.assert_almost_equal(angle, 0)
    # Test zeros quaternion
    axis, angle = br.quaternion.to_axis_angle([0, 0, 0, 0])
    np.testing.assert_allclose(axis, [1, 0, 0])
    np.testing.assert_almost_equal(angle, 0)
    # Test invalid quaternion
    axis, angle = br.quaternion.to_axis_angle([float('inf'), 0, 0, 0])
    np.testing.assert_allclose(axis, [1, 0, 0])
    self.assertTrue(math.isnan(angle))


  def test_to_euler(self):
    q = [0.70105738, 0.43045933,  0.56098553, -0.09229596]
    ai, aj, ak = br.quaternion.to_euler(q, 'sxyz')
    np.testing.assert_allclose([ai, aj, ak], [np.pi/2, np.pi/3, np.pi/4,])

  def test_to_transform(self):
    T0 = br.quaternion.to_transform([0.99810947, 0.06146124, 0, 0])
    T1 = br.axis_angle.to_transform([1, 0, 0], 0.123)
    np.testing.assert_allclose(T0, T1)
    T = br.quaternion.to_transform([1, 0, 0, 0])
    np.testing.assert_allclose(T, np.identity(4))
    T = br.quaternion.to_transform([0, 1, 0, 0])
    np.testing.assert_allclose(T, np.diag([1, -1, -1, 1]))
    # Test zeros
    T = br.quaternion.to_transform(np.zeros(4))
    self.assertTrue(br.transform.are_equal(T, np.eye(4)))
