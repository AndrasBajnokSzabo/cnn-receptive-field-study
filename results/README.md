# Results

Running `python run_experiments.py` writes fresh experiment outputs to `results/generated/` by default:

```text
results/generated/
├── results_all.csv
├── best_results_by_dataset_and_model.csv
├── histories/
└── plots/
```

`best_results_by_dataset_and_model.csv` selects the best kernel configuration **using validation accuracy**, then reports its held-out test metrics. The test split is therefore not used for model/configuration selection.

`reported_findings.md` contains the headline values stated in the original presentation. The complete original 90-run CSV was not part of the provided source material, so missing measurements are intentionally not fabricated; rerunning the experiment grid generates a fresh complete table.
