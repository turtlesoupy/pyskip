from unittest import TestCase

import skimpy

class TestSimple(TestCase):
    def test_tensor(self):
        tensor = skimpy.Tensor()
        self.assertIsNotNone(tensor)