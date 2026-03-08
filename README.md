# CUMULUS: Robust Preprocessing Pipelines for Eye-Tracking Data

CUMULUS is a controlled benchmarking framework for evaluating preprocessing pipelines for eye-tracking time series under injected missingness and outlier corruption.

The benchmark evaluates three preprocessing stages:

1. missing-value imputation
2. outlier handling
3. normalization

It produces Excel workbooks with raw and aggregated results for each dataset.

## Implemented methods

### Imputation
- Mean imputation
- LOCF (last observation carried forward with backward fill)
- KNN imputation (`k=5`)
- Optional: GPR
- Optional: MICE

### Outlier handling
- Z-score filtering
- MAD filtering
- Isolation Forest

### Normalization
- Min-Max scaling
- Z-score standardization
- Robust scaling

## Expected input format

The script expects CSV files containing the following canonical columns:

- `Gaze X`
- `Gaze Y`
- `ET_PupilLeft`
- `ET_PupilRight`

If your dataset uses different column names, edit the `COLUMN_ALIASES` dictionary in `cumulus_benchmark.py`.

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
