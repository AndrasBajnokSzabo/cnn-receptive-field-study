"""Plotting and result-summary helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_lineplot(df: pd.DataFrame, x: str, y: str, hue: str, title: str, out_path: Path):
    """Save a simple line plot grouped by ``hue``."""
    plt.figure(figsize=(10, 6))
    for label, part in df.groupby(hue):
        part = part.sort_values(x)
        plt.plot(part[x], part[y], marker="o", label=str(label), linewidth=1.5, markersize=4)
    plt.xlabel(x.replace("_", " ").title())
    plt.ylabel(y.replace("_", " ").title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def make_all_plots(results_df: pd.DataFrame, out_dir: str):
    """Create plots and a validation-selected best-configuration table.

    The test split is never used for model/configuration selection. For each
    dataset/model pair, the selected row is the one with the highest validation
    accuracy; its test metrics are then reported for final evaluation.
    """
    out_dir = Path(out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, df_ds in results_df.groupby("dataset"):
        specs = [
            ("test_accuracy", "accuracy_vs_kernel"),
            ("test_macro_f1", "f1_vs_kernel"),
            ("total_train_time_sec", "time_vs_kernel"),
            ("num_params", "params_vs_kernel"),
        ]
        for metric, suffix in specs:
            save_lineplot(
                df_ds,
                x="kernel_size",
                y=metric,
                hue="model_type",
                title=f"{metric.replace('_', ' ').title()} vs. kernel size - {dataset_name}",
                out_path=plots_dir / f"{dataset_name}_{suffix}.png",
            )

    for model_type, df_model in results_df.groupby("model_type"):
        save_lineplot(
            df_model,
            x="kernel_size",
            y="test_accuracy",
            hue="dataset",
            title=f"Test accuracy vs. kernel size - {model_type}",
            out_path=plots_dir / f"{model_type}_datasets_accuracy_vs_kernel.png",
        )

    selected_rows = []
    for _, part in results_df.groupby(["dataset", "model_type"]):
        # Select on validation performance, never on the held-out test set.
        selected_rows.append(part.sort_values("best_val_accuracy", ascending=False).iloc[0])

    pd.DataFrame(selected_rows).to_csv(
        out_dir / "best_results_by_dataset_and_model.csv",
        index=False,
    )
