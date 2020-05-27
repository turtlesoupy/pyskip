from unittest import TestCase

import numpy as np
import skimpy
import skimpy.functional as F


class TestConvolve(TestCase):
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

  def test_conv_2d(self):
    disc = TestConvolve.make_disc()
    sobel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1],
    ])

    edge = F.conv_2d(
        disc,
        skimpy.Tensor.from_numpy(sobel),
        padding = 1,
    )

    self.assertEqual(F.sum(edge.abs()), 32768)
