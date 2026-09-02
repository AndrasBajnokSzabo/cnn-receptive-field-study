"""Lightweight shape and receptive-field smoke tests."""

import unittest

import torch

from src.models import MODEL_REGISTRY, convolutional_receptive_field


class ModelSmokeTests(unittest.TestCase):
    def test_mnist_output_shapes(self):
        x = torch.randn(2, 1, 28, 28)
        for name, model_cls in MODEL_REGISTRY.items():
            for kernel_size in (1, 2, 7, 28):
                with self.subTest(model=name, kernel=kernel_size):
                    model = model_cls(kernel_size=kernel_size, num_classes=10, in_channels=1)
                    self.assertEqual(tuple(model(x).shape), (2, 10))

    def test_cifar_output_shapes(self):
        x = torch.randn(2, 3, 32, 32)
        for name, model_cls in MODEL_REGISTRY.items():
            for kernel_size in (1, 2, 7, 28):
                with self.subTest(model=name, kernel=kernel_size):
                    model = model_cls(kernel_size=kernel_size, num_classes=100, in_channels=3)
                    self.assertEqual(tuple(model(x).shape), (2, 100))

    def test_reported_convolutional_receptive_field(self):
        self.assertEqual(convolutional_receptive_field([7]), 7)
        self.assertEqual(convolutional_receptive_field([7, 7, 7]), 19)


if __name__ == "__main__":
    unittest.main()
