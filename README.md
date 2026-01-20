# CUMULUS — Cleaning, Unifying, and Modeling Unstable Eye-tracking Signals

Code repository for the paper:

**CUMULUS: Cleaning, Unifying, and Modeling Unstable Eye-tracking Signals**  
Jennifer Landes, Meike Klettke, Sonja Koeppl

This repository provides the implementation used to benchmark common preprocessing stages for eye-tracking time series:
missing-value imputation, outlier handling, and normalization (pipeline: impute → outlier → normalize).

## What this repo contains

- `notebooks/cumulus_clean.ipynb`  
  Main notebook to reproduce the preprocessing benchmark and export results (CSV) and optional figures.

- `results/` (generated)  
  Output tables (e.g., imputation/outlier/normalization summaries).

- `figures/` (optional, generated)  
  Plots used for inspection and (optionally) paper figures.

## Requirements

- Python 3.10+ recommended (3.9 may work, but not guaranteed)

## Setup

```bash
git clone <YOUR_REPO_URL>
cd <YOUR_REPO_FOLDER>

python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
