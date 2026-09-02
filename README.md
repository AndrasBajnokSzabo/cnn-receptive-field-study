# CNN Receptive Field Study

A compact empirical study of how **convolutional kernel size and receptive field** affect image-classification performance, computational cost, and parameter efficiency across several CNN architectures.

The project compares three datasets, three CNN designs, and ten kernel sizes for a total of **90 controlled training configurations**.

## Study design

- **Datasets:** MNIST, CIFAR-10, CIFAR-100
- **Architectures:** SimpleCNN, DeepCNN, DepthwiseCNN
- **Kernel sizes:** `1, 2, 3, 5, 7, 9, 11, 15, 21, 28`
- **Total configurations:** 3 datasets × 3 architectures × 10 kernels = **90 training runs**
- **Optimizer:** AdamW
- **Loss:** CrossEntropyLoss
- **Default training length:** 5 epochs
- **Metrics:** validation/test accuracy, macro-F1, parameter count, training time, and theoretical convolution-stack receptive field

The models use **Global Average Pooling (GAP)**, so the same implementation can handle both 28×28 MNIST and 32×32 CIFAR images.

## Research question

> How do larger convolution kernels and receptive fields trade off classification performance against parameter count and training cost, and when are depthwise-separable convolutions a better choice than standard convolutions?

## Key findings

The original experiment reported several clear trends:

- **MNIST:** DeepCNN and DepthwiseCNN reach roughly 97-99% from `k=3`, while the shallow SimpleCNN benefits strongly from larger kernels and peaks around `k=21`.
- **CIFAR-10:** DeepCNN peaks at roughly 67% around `k=7`; DepthwiseCNN reaches roughly 61% around `k=9-11`.
- **CIFAR-100:** DeepCNN peaks at roughly 30% around `k=9`, and performance generally declines for very large kernels.
- **Efficiency:** at `k=28`, the study reports roughly **8M parameters for DeepCNN** versus about **130k for DepthwiseCNN**, while measured training time was about **1250 s vs. 350 s**, respectively.
- **Main takeaway:** a larger receptive field does not automatically improve accuracy. Architecture depth and parameter-efficient spatial filtering matter more than simply increasing kernel size.

These are presentation-level summary values, not reconstructed raw measurements. A fresh `results_all.csv` is produced by rerunning the experiment grid.

![CIFAR-10 results](figures/cifar10_results.png)

## Architectures

### SimpleCNN

A shallow baseline with one kernel-size-dependent convolution, followed by MaxPool, GAP, and a small fully connected classifier.

### DeepCNN

Three standard convolution layers with channel widths `32 → 64 → 128`. It has the highest capacity, but parameter count and computation increase rapidly with kernel size.

### DepthwiseCNN

Three depthwise-separable blocks. Spatial filtering is performed independently per channel and followed by a `1×1` pointwise convolution for channel mixing, making large kernels substantially more parameter-efficient.

## Efficiency comparison

![Training time vs. kernel size](figures/training_time.png)

![Parameter count vs. kernel size](figures/parameter_count.png)

Wall-clock timings are hardware-dependent. Parameter-count trends follow directly from the model definitions.

## Repository structure

```text
cnn-receptive-field-study/
├── README.md
├── LICENSE
├── requirements.txt
├── run_experiments.py
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   ├── plotting.py
│   └── training.py
├── tests/
│   └── test_models.py
├── results/
│   ├── README.md
│   └── reported_findings.md
├── figures/
│   ├── mnist_results.png
│   ├── cifar10_results.png
│   ├── cifar100_results.png
│   ├── training_time.png
│   ├── parameter_count.png
│   └── summary.png
└── presentation/
    ├── receptive_field_study_en.pdf
    └── receptive_field_study_en.pptx
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

## Running the experiments

### Full 90-run experiment

```bash
python run_experiments.py --epochs 5
```

### Quick smoke run

```bash
python run_experiments.py \
  --epochs 1 \
  --subset_train 5000 \
  --subset_test 1000 \
  --max_kernel 7
```

### One dataset and one model

```bash
python run_experiments.py \
  --datasets MNIST \
  --models SimpleCNN \
  --epochs 5
```

### CPU only

```bash
python run_experiments.py --cpu --epochs 1 --max_kernel 7
```

## Testing

A lightweight smoke-test suite checks output tensor shapes for MNIST/CIFAR inputs, including even kernels and the largest `28×28` kernel, and verifies the reported convolution-stack receptive-field calculation.

```bash
python -m unittest discover -s tests
```

## Generated outputs

Fresh experiment outputs are written to `results/generated/` by default:

```text
results/generated/
├── results_all.csv
├── best_results_by_dataset_and_model.csv
├── histories/
└── plots/
```

Each experiment-level row records the validation/test metrics, parameter count, receptive field, timing, training hyperparameters, and random seed.

The summary table selects the best kernel configuration for each dataset/model pair using **validation accuracy**. The held-out test split is used only for final reporting and is not used for configuration selection.

## Reproducibility details

- Python, NumPy, and PyTorch random seeds are fixed.
- The same deterministic train/validation split is reused across configurations within a dataset.
- Test data is kept separate from model selection.
- Explicit asymmetric SAME padding supports both odd and even kernels without changing feature-map dimensions.
- CUDA is synchronized around timed training sections so GPU wall-clock measurements are not artificially underestimated by asynchronous execution.
- Result rows store the seed, batch size, learning rate, and weight decay.

## Receptive-field convention

For consistency with the original experiment and presentation, `theoretical_receptive_field` refers to the **stride-1 convolution stack only**:

- SimpleCNN: `RF = k`
- DeepCNN: `RF = 1 + 3(k - 1)`
- DepthwiseCNN: `RF = 1 + 3(k - 1)`

Pooling layers are intentionally excluded from this reported value. This convention is explicit in the refactored implementation to avoid ambiguity.

## Limitations

This is a small controlled study rather than a state-of-the-art image-classification benchmark.

- The reported experiment uses a **single random seed**, so small differences should not be treated as statistically robust.
- Training is limited to **5 epochs** to keep the 90-run sweep computationally manageable.
- There is no extensive hyperparameter tuning or data augmentation.
- Wall-clock training times depend on the hardware and software environment.
- The study isolates kernel-size / architecture trade-offs; it is not designed to maximize CIFAR accuracy.
- The original complete 90-run raw CSV was not included in the provided source material, so the repository does not fabricate missing measurements.

A stronger follow-up would repeat each configuration with multiple seeds and report mean ± standard deviation.

## Presentation

An English, portfolio-ready project presentation is included in both formats:

- `presentation/receptive_field_study_en.pdf` - easy to preview directly on GitHub
- `presentation/receptive_field_study_en.pptx` - editable PowerPoint version

The presentation is a useful **supplement** for quickly communicating the experiment setup and conclusions; the README and source code remain the primary GitHub content.

![Summary slide](figures/summary.png)

## Project scope

This repository is intentionally a **small experimental ML project**, not a production package. Its purpose is to demonstrate controlled experimentation, architecture comparison, metric reporting, reproducibility practices, and analysis of accuracy/efficiency trade-offs.

## License

MIT License.
