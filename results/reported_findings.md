# Reported findings

These are the headline observations reported in the original project presentation. They are included for portfolio context; they are **not** reconstructed raw measurements.

- **MNIST:** DeepCNN and DepthwiseCNN reach roughly 97-99% accuracy from about `k=3` onward. SimpleCNN improves from about 26% at `k=1` to a peak near 89% at `k=21`.
- **CIFAR-10:** DeepCNN peaks at roughly 67% around `k=7`. DepthwiseCNN reaches roughly 61% around `k=9-11`, then declines. SimpleCNN remains around 31-45%.
- **CIFAR-100:** DeepCNN peaks around 30% near `k=9`; performance generally falls for kernels larger than about 9. SimpleCNN stays around 10-15%.
- **Training time:** at `k=28`, DeepCNN takes roughly 1250 s, while DepthwiseCNN takes roughly 350 s (about 3.5x faster). SimpleCNN remains around 50 s in the reported setup.
- **Parameter count:** at `k=28`, DeepCNN has roughly 8M parameters, DepthwiseCNN roughly 130k (about 60x fewer), and SimpleCNN roughly 90k.

The study is intentionally a compact controlled comparison rather than a state-of-the-art image-classification benchmark.
