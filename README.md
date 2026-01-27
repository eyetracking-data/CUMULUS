# CUMULUS – Robust Preprocessing Pipelines for Eye-Tracking Data

This repository provides the **CUMULUS benchmark**, a controlled evaluation framework for preprocessing
eye-tracking time series as described in the accompanying paper  
*“Robust Preprocessing Pipelines for Eye-Tracking Data in Work-Related HCI”*.

CUMULUS evaluates complete preprocessing pipelines across three stages:
missing-value imputation, outlier handling, and normalization, under systematically injected corruptions.
The focus is on reconstruction accuracy and distributional stability rather than downstream classifier performance.

---

## Repository Structure

```text
CUMULUS/
├── CUMULUS.ipynb   # Main benchmark notebook
├── data/
│   └── raw/                      # Raw eye-tracking CSV files (not included)
├── results/                      # Generated result tables (CSV)
├── figures/                      # Generated figures (optional)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Data

The benchmark expects **task-level eye-tracking CSV files** in the directory:

```text
data/raw/
```

In the paper, we use **125 CSV files** corresponding to  
25 participants × 5 task segments.

Due to data protection constraints, raw data are **not included** in this repository.
The dataset used in the paper is available separately at:

https://github.com/eyetracking-data/eyetracking_Cheating

All CSV files should share a common schema and contain, at minimum, the following features:

- `Gaze X`
- `Gaze Y`
- `ET_PupilLeft`
- `ET_PupilRight`

---

## Running the Benchmark

1. Install dependencies (recommended: use a virtual environment):

```bash
pip install -r requirements.txt
```

2. Place all raw CSV files into:

```text
data/raw/
```

3. Run the benchmark notebook:

```text
CUMULUS_paper_aligned.ipynb
```

The notebook:
- iterates over **all CSV files** in `data/raw/`
- injects controlled corruption (MCAR missingness, spike-like outliers)
- evaluates preprocessing methods **per file**
- aggregates metrics **across files** (median)
- writes result tables to `results/`

---

## Output

After execution, the following files are generated:

- `results/imputation_results_raw.csv`  
  Per-file evaluation results.

- `results/imputation_results_summary.csv`  
  Aggregated results used for tables and figures in the paper.

Figures shown in the paper are generated from these aggregated tables; individual plots shown in the manuscript are representative examples.

---

## Reproducibility Notes

- Corruption mechanisms (missingness and outliers) are injected under **fixed random seeds** to ensure comparability across methods.
- Hyperparameters (e.g., KNN with *k = 5*, Isolation Forest contamination = 0.05) are fixed to reflect realistic practitioner defaults and to avoid tuning on evaluation data.
- Runtime depends on the number and length of CSV files; the benchmark is designed for **methodological transparency**, not minimal execution time.

---

## Scope and Limitations

CUMULUS provides **robust default guidance under controlled conditions**
(MCAR missingness and spike-like outliers).
Results may differ under structured missingness (MAR/MNAR), gradual artifacts (e.g., drift),
or geometry-dependent analyses (AOI/ROI), as discussed in the paper.

---

## License and Citation

Please cite the accompanying paper when using this benchmark.
