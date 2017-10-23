#! /usr/bin/env python
import unittest
import numpy as np
# Tested package
import baldor as br


class TestModule(unittest.TestCase):
  def test_to_euler(self):
    ai, aj, ak = br.axis_angle.to_euler([1, 0, 0], 0, axes='sxyz')
    np.testing.assert_allclose([ai, aj, ak], 0, rtol=1e-7, atol=1e-8)
    ai, aj, ak = br.axis_angle.to_euler([0, 1, 0], 1.5, axes='sxyz')
    np.testing.assert_allclose([ai, aj, ak], [0, 1.5, 0], rtol=1e-7, atol=1e-8)

  def test_to_quaternion(self):
    q = br.axis_angle.to_quaternion([1., 0., 0.], np.pi, isunit=True)
    np.testing.assert_allclose(q, [0., 1., 0., 0.], rtol=1e-7, atol=1e-8)
    q = br.axis_angle.to_quaternion([2, 0, 0], np.pi, isunit=False)
    np.testing.assert_allclose(q, [0., 1., 0., 0.], rtol=1e-7, atol=1e-8)

  def test_to_transform(self):
    # Know transform
    T = br.axis_angle.to_transform([0, 0, 1], np.pi/2, [1, 0, 0])
    np.testing.assert_allclose(np.dot(T, [0, 0, 0, 1]), [1, -1, 0, 1])
    # Random axis angle and point
    angle = (np.random.sample() - 0.5) * (2*np.pi)
    axis = np.random.sample(3) - 0.5
    point = np.random.sample(3) - 0.5
    T0 = br.axis_angle.to_transform(axis, angle, point)
    T1 = br.axis_angle.to_transform(axis, angle-2*np.pi, point)
    self.assertTrue(br.transform.are_equal(T0, T1))
    T0 = br.axis_angle.to_transform(axis, angle, point)
    T1 = br.axis_angle.to_transform(-axis, -angle, point)
    self.assertTrue(br.transform.are_equal(T0, T1))
    T = br.axis_angle.to_transform(axis, np.pi*2)
    np.testing.assert_allclose(T, np.eye(4), rtol=1e-7, atol=1e-8)
    T = br.axis_angle.to_transform(axis, np.pi/2., point)
    np.testing.assert_allclose(np.trace(T), 2., rtol=1e-7, atol=1e-8)
