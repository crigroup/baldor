#! /usr/bin/env python
import unittest
import numpy as np
# Tested package
import baldor as br


class TestModule(unittest.TestCase):
  def test_are_equal(self):
    T0 = np.diag([1, 1, 1, 1])
    T1 = np.eye(4)
    self.assertTrue(br.quaternion.are_equal(T0, T1))
    T2 = br.quaternion.to_transform([1, 0, 0, 0])
    T3 = br.quaternion.to_transform([-1, 0, 0, 0])
    self.assertTrue(br.quaternion.are_equal(T2, T3))

  def test_inverse(self):
    q = br.quaternion.random()
    T = br.quaternion.to_transform(q)
    Tinv = br.transform.inverse(T)
    np.testing.assert_allclose(np.dot(T, Tinv), np.eye(4), rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(Tinv, np.linalg.inv(T), rtol=1e-5, atol=1e-8)

  def test_to_axis_angle(self):
    axis = np.random.sample(3) - 0.5
    angle = (np.random.sample(1) - 0.5) * (2*np.pi)
    point = np.random.sample(3) - 0.5
    T0 = br.axis_angle.to_transform(axis, angle, point)
    axis, angle, point = br.transform.to_axis_angle(T0)
    T1 = br.axis_angle.to_transform(axis, angle, point)
    self.assertTrue(br.transform.are_equal(T0, T1))

  def test_to_euler(self):
    T0 = br.euler.to_transform(1, 2, 3, 'syxz')
    al, be, ga = br.transform.to_euler(T0, 'syxz')
    T1 = br.euler.to_transform(al, be, ga, 'syxz')
    np.testing.assert_allclose(T0, T1)

  def test_to_transform(self):
    q = br.transform.to_quaternion(np.identity(4), isprecise=True)
    np.testing.assert_allclose(q, [1, 0, 0, 0])
    q = br.transform.to_quaternion(np.diag([1, -1, -1, 1]))
    self.assertTrue( np.allclose(q, [0, 1, 0, 0]) or
                                                np.allclose(q, [0, -1, 0, 0]) )
    T = br.axis_angle.to_transform((1, 2, 3), 0.123)
    q = br.transform.to_quaternion(T, True)
    np.testing.assert_allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786],
                                                          rtol=1e-5, atol=1e-8)
