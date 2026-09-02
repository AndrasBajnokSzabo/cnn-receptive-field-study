"""Run the full kernel-size × architecture × dataset experiment grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.data import make_loaders
from src.models import MODEL_REGISTRY, convolutional_receptive_field
from src.plotting import make_all_plots
from src.training import ExperimentResult, count_parameters, evaluate, set_seed, train_one_model

DEFAULT_KERNELS = [1, 2, 3, 5, 7, 9, 11, 15, 21, 28]


def run_experiments(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    histories_dir = out_dir / "histories"
    histories_dir.mkdir(parents=True, exist_ok=True)

    datasets_to_run = [x.strip() for x in args.datasets.split(",") if x.strip()]
    models_to_run = [x.strip() for x in args.models.split(",") if x.strip()]
    kernel_sizes = [k for k in DEFAULT_KERNELS if args.min_kernel <= k <= args.max_kernel]
    if not kernel_sizes:
        raise ValueError("No kernel sizes remain after applying min/max filters.")

    invalid_models = [m for m in models_to_run if m not in MODEL_REGISTRY]
    if invalid_models:
        raise ValueError(f"Unknown model(s): {invalid_models}. Available: {list(MODEL_REGISTRY)}")

    all_results = []
    for dataset_name in datasets_to_run:
        print(f"\n{'=' * 80}\nDataset: {dataset_name}\n{'=' * 80}")
        loaders = make_loaders(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            num_workers=args.num_workers,
            subset_train=args.subset_train,
            subset_test=args.subset_test,
            seed=args.seed,
            pin_memory=device.type == "cuda",
        )
        train_loader, val_loader, test_loader, num_classes, in_channels = loaders

        for model_type in models_to_run:
            for kernel_size in kernel_sizes:
                print(f"Run: dataset={dataset_name}, model={model_type}, kernel={kernel_size}x{kernel_size}")
                model = MODEL_REGISTRY[model_type](
                    kernel_size=kernel_size,
                    num_classes=num_classes,
                    in_channels=in_channels,
                ).to(device)

                n_params = count_parameters(model)
                rf = convolutional_receptive_field(model.rf_kernel_sizes)
                model, history, total_time = train_one_model(
                    model, train_loader, val_loader, device,
                    args.epochs, args.lr, args.weight_decay,
                )

                hist_df = pd.DataFrame(history)
                hist_df.to_csv(histories_dir / f"history_{dataset_name}_{model_type}_k{kernel_size}.csv", index=False)
                best_idx = hist_df["val_accuracy"].idxmax()
                test_acc, test_f1 = evaluate(model, test_loader, device)

                result = ExperimentResult(
                    dataset=dataset_name,
                    model_type=model_type,
                    kernel_size=kernel_size,
                    theoretical_receptive_field=rf,
                    num_params=n_params,
                    best_val_accuracy=float(hist_df.loc[best_idx, "val_accuracy"]),
                    best_val_macro_f1=float(hist_df.loc[best_idx, "val_macro_f1"]),
                    test_accuracy=float(test_acc),
                    test_macro_f1=float(test_f1),
                    total_train_time_sec=float(total_time),
                    avg_epoch_time_sec=float(total_time / args.epochs),
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    weight_decay=args.weight_decay,
                    seed=args.seed,
                )
                all_results.append(result.to_dict())
                pd.DataFrame(all_results).to_csv(out_dir / "results_all.csv", index=False)
                print(f"  test_acc={test_acc:.4f}, test_f1={test_f1:.4f}, params={n_params:,}, RF={rf}, time={total_time:.1f}s")

    results_df = pd.DataFrame(all_results)
    make_all_plots(results_df, str(out_dir))
    print(f"\nDone. Results: {out_dir / 'results_all.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--out_dir", default="./results/generated")
    parser.add_argument("--datasets", default="MNIST,CIFAR10,CIFAR100")
    parser.add_argument("--models", default="SimpleCNN,DeepCNN,DepthwiseCNN")
    parser.add_argument("--min_kernel", type=int, default=1)
    parser.add_argument("--max_kernel", type=int, default=28)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--subset_train", type=int, default=0)
    parser.add_argument("--subset_test", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiments(parse_args())
