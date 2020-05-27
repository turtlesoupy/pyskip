from unittest import TestCase

import numpy as np
import skimpy
import skimpy.reduce as R


class TestReduce(TestCase):
    @classmethod
    def make_disc(cls):
        D = 4096
        R = D // 4
        disc = skimpy.Tensor((D, D), 0)
        for y in range(D):
            discriminant = R**2 - (y - D // 2 + 0.5)**2
            if discriminant < 0:
                continue
            x_0 = int(D // 2 - 0.5 - discriminant**0.5)
            x_1 = int(D // 2 - 0.5 + discriminant**0.5)
            disc[x_0:x_1, y] = 1
        return disc

    def test_sum(self):
        disc = TestReduce.make_disc()
        self.assertEqual(R.sum(disc), 3294288)
        self.assertEqual(R.sum(disc, keepdims=True).item(), 3294288)
        self.assertEqual(R.sum(disc), np.sum(disc.to_numpy()))
