"""Dataset and DataLoader utilities."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_dataset(dataset_name: str, data_dir: str):
    name = dataset_name.lower()
    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        train = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        return train, test, 10, 1

    if name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        return train, test, 10, 3

    if name == "cifar100":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        train = datasets.CIFAR100(data_dir, train=True, download=True, transform=transform)
        test = datasets.CIFAR100(data_dir, train=False, download=True, transform=transform)
        return train, test, 100, 3

    raise ValueError(f"Unknown dataset: {dataset_name}")


def _take_subset(dataset, n: int, seed: int):
    if n <= 0:
        return dataset
    if n > len(dataset):
        raise ValueError(f"Requested subset size {n} exceeds dataset size {len(dataset)}.")
    subset, _ = random_split(
        dataset,
        [n, len(dataset) - n],
        generator=torch.Generator().manual_seed(seed),
    )
    return subset


def make_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    val_ratio: float,
    num_workers: int,
    subset_train: int,
    subset_test: int,
    seed: int,
    pin_memory: bool,
):
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    train_full, test_set, num_classes, in_channels = get_dataset(dataset_name, data_dir)
    train_full = _take_subset(train_full, subset_train, seed)
    test_set = _take_subset(test_set, subset_test, seed)

    val_size = max(1, int(len(train_full) * val_ratio))
    train_size = len(train_full) - val_size
    if train_size < 1:
        raise ValueError("Training subset is too small after creating the validation split.")

    train_set, val_set = random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_set, shuffle=True, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader, num_classes, in_channels
