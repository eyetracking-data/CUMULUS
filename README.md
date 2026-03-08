# CUMULUS: Robust Preprocessing Pipelines for Eye-Tracking Data

CUMULUS is a controlled benchmarking framework for evaluating
preprocessing pipelines for eye‑tracking time series under injected
missingness and outlier corruption.

The benchmark evaluates three preprocessing stages:

1.  Missing‑value imputation\
2.  Outlier handling\
3.  Normalization

The framework produces Excel workbooks with raw and aggregated results
for each dataset.

------------------------------------------------------------------------

## Implemented Methods

### Imputation

-   Mean imputation
-   LOCF (last observation carried forward)
-   KNN imputation (k=5)
-   Optional: Gaussian Process Regression (GPR)
-   Optional: Multiple Imputation by Chained Equations (MICE)

### Outlier Handling

-   Z‑score filtering
-   MAD filtering (Median Absolute Deviation)
-   Isolation Forest

### Normalization

-   Min--Max scaling
-   Z‑score standardization
-   Robust scaling (median / IQR)

------------------------------------------------------------------------

## Expected Input Format

The script expects CSV files containing the following canonical columns:

-   `Gaze X`
-   `Gaze Y`
-   `ET_PupilLeft`
-   `ET_PupilRight`

If your dataset uses different column names, adjust the `COLUMN_ALIASES`
dictionary in `cumulus_benchmark.py`.

------------------------------------------------------------------------

## Installation

Create a Python environment and install dependencies:

``` bash
pip install -r requirements.txt
```

Dependencies include:

-   numpy
-   pandas
-   scipy
-   scikit‑learn
-   openpyxl

------------------------------------------------------------------------

## Usage

Run the benchmark on one or more dataset directories:

``` bash
python cumulus.py   --datasets /path/to/dataset1 /path/to/dataset2   --output-dir /path/to/results   --recursive
```

### Optional Flags

Enable computationally expensive imputation methods:

``` bash
--enable-gpr
--enable-mice
```

Specify a custom random seed:

``` bash
--random-seed 42
```

------------------------------------------------------------------------

## Output

For each dataset the benchmark generates one Excel workbook:

    CUMULUS_<dataset_name>.xlsx

Each workbook contains:

-   `run_log`
-   `imputation_raw`
-   `imputation_summary`
-   `outlier_raw`
-   `outlier_summary`
-   `normalization_raw`
-   `normalization_summary`

These sheets allow full traceability from aggregated statistics back to
individual files.

------------------------------------------------------------------------

## Reproducibility

All corruption injections use fixed random seeds by default so that the
same corruption patterns can be reproduced across runs.

------------------------------------------------------------------------

## Citation

If you use this code in academic work, please cite the associated
**CUMULUS paper** on robust preprocessing pipelines for eye‑tracking
data.
