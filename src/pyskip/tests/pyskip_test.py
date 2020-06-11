from unittest import TestCase

import numpy as np
import pyskip


class TestTensor(TestCase):
    @classmethod
    def make_disc(cls):
        D = 4096
        R = D // 4
        disc = pyskip.Tensor((D, D), 0)
        for y in range(D):
            discriminant = R**2 - (y - D // 2 + 0.5)**2
            if discriminant < 0:
                continue
            x_0 = int(D // 2 - 0.5 - discriminant**0.5)
            x_1 = int(D // 2 - 0.5 + discriminant**0.5)
            disc[x_0:x_1, y] = 1
        return disc

    def test_tensor_1d(self):
        tensor = pyskip.Tensor((100, ), 0)
        self.assertIsNotNone(tensor)

    def test_tensor_2d(self):
        tensor = pyskip.Tensor((100, 100), 0)
        self.assertIsNotNone(tensor)

    def test_tensor_3d(self):
        tensor = pyskip.Tensor((100, 100, 100), 0)
        self.assertIsNotNone(tensor)

    def test_slicing(self):
        tensor = pyskip.Tensor((3, 3), 0)
        tensor[1, 0:2] = 1
        tensor[0:2, 0] = 2
        tensor[2:3, 2:3] = 3

        expected = np.array([[2, 2, 0], [0, 1, 0], [0, 0, 3]], dtype=int)
        self.assertTrue(np.all(tensor.to_numpy() == np.array(expected)))

    def test_operations(self):
        tensor = pyskip.Tensor(3, 1)
        tensor *= 3
        tensor[0] = 2 * tensor[1]
        tensor[0] += tensor[2] * 3
        tensor[1] += tensor[0] * tensor[2]
        self.assertEqual(tensor[0], 15)
        self.assertEqual(tensor[1], 48)
        self.assertEqual(tensor[2], 3)

        tensor = pyskip.Tensor(shape=(4, 4), val=0)
        tensor[0::2, 0::2] = 1
        tensor[1::2, 0::2] = 2
        tensor[0::2, 1::2] = 3
        tensor[1::2, 1::2] = 4
        expected = np.array([[1, 2, 1, 2], [3, 4, 3, 4], [1, 2, 1, 2], [3, 4, 3, 4]], dtype=int)
        self.assertTrue(np.all(tensor.to_numpy() == np.array(expected)))

        expected = np.array([2, 2, 2, 2], dtype=int)
        self.assertTrue(np.all(tensor[1:4:2, 0:4:2].to_numpy().flatten() == np.array(expected)))
