from unittest import TestCase

import numpy as np
import skimpy


class TestTensor(TestCase):
  def test_tensor_1d(self):
    tensor = skimpy.Tensor((100, ), 0)
    self.assertIsNotNone(tensor)

  def test_tensor_2d(self):
    tensor = skimpy.Tensor((100, 100), 0)
    self.assertIsNotNone(tensor)

  def test_tensor_3d(self):
    tensor = skimpy.Tensor((100, 100, 100), 0)
    self.assertIsNotNone(tensor)

  def test_slicing(self):
    tensor = skimpy.Tensor((3, 3), 0)
    tensor[1, 0:2] = 1
    tensor[0:2, 0] = 2
    tensor[2:3, 2:3] = 3

    expected = np.array([[2, 2, 0], [0, 1, 0], [0, 0, 3]], dtype = int)
    self.assertTrue(np.all(tensor.to_numpy() == np.array(expected)))

  def test_operations(self):
    tensor = skimpy.Tensor(3, 1)
    tensor *= 3
    tensor[0] = 2 * tensor[1]
    tensor[0] += tensor[2] * 3
    tensor[1] += tensor[0] * tensor[2]
    self.assertEqual(tensor[0], 15)
    self.assertEqual(tensor[1], 48)
    self.assertEqual(tensor[2], 3)
