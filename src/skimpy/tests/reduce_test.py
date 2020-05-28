from unittest import TestCase

import numpy as np
from skimpy import Tensor, reduce as R


class TestReduce(TestCase):
    @classmethod
    def make_disc(cls):
        D = 4096
        R = D // 4
        disc = Tensor((D, D), 0)
        for y in range(D):
            discriminant = R**2 - (y - D // 2 + 0.5)**2
            if discriminant < 0:
                continue
            x_0 = int(D // 2 - 0.5 - discriminant**0.5)
            x_1 = int(D // 2 - 0.5 + discriminant**0.5)
            disc[x_0:x_1, y] = 1
        return disc

    def test_sum_simple(self):
        self.assertEqual(R.sum(Tensor.from_list([])), 0)
        self.assertEqual(R.sum(Tensor.from_list([2])), 2)
        self.assertEqual(R.sum(Tensor.from_list([2, -3])), -1)
        self.assertEqual(R.sum(Tensor.from_list([2, -3, 4])), 3)
        self.assertEqual(R.sum(Tensor.from_list([1, 2, 3, 4, 5, 6, 7, 8])), 36)

    def test_sum_disc(self):
        disc = TestReduce.make_disc()
        self.assertEqual(R.sum(disc), 3294288)
        self.assertEqual(R.sum(disc, keepdims=True).item(), 3294288)
        self.assertEqual(R.sum(disc), np.sum(disc.to_numpy()))
