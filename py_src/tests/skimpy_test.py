from unittest import TestCase

import skimpy

class TestTensor(TestCase):
    def test_tensor_1d(self):
        tensor = skimpy.Tensor((100,), 0)
        self.assertIsNotNone(tensor)

    def test_tensor_2d(self):
        tensor = skimpy.Tensor((100, 100), 0)
        self.assertIsNotNone(tensor)

    def test_tensor_3d(self):
        tensor = skimpy.Tensor((100, 100, 100), 0)
        self.assertIsNotNone(tensor)