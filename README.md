# CUMULUS: Robust Preprocessing Pipelines for Eye-Tracking Data

This repository contains a compact, paper-aligned reproduction workflow for CUMULUS.

## Repository structure

```text
CUMULUS/
├─ CUMULUS_Reproduction.ipynb
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ results/
   ├─ tables/
   │  ├─ paper_table2_reference.csv
   │  └─ paper_table3_reference.csv
   └─ figures/
```

The notebook implements the paper's three-stage design space:

**Imputation → Outlier handling → Normalization**

- Imputation: Mean, LOCF, KNN (`k=5`)
- Outlier handling: Z-score (`|z| > 3`), MAD, Isolation Forest
- Normalization: Min-Max, Z-score, RobustScaler

This yields **27 complete pipeline combinations**.

The controlled corruption levels are **5%, 10%, 15%, and 20%**, using fixed random seeds.

## What the notebook produces

The notebook writes reproducible outputs to `results/tables/` and figures to `results/figures/`.

Main generated tables include:

- run log
- stage-level imputation summaries
- stage-level outlier summaries
- normalization summaries
- raw metrics for all 27 complete pipelines
- dataset-level pipeline metric vectors
- metric-specific ranks
- the metric vector of the paper-reported `KNN → Isolation Forest → Min-Max` pipeline
- a direct comparison between recomputed stage values and the published Table 2 values

The two files beginning with `paper_` are **literal reference transcriptions of the published paper**, not recomputed results.

## Data

The raw datasets are not included in the repository.

The notebook is preconfigured for the local Dropbox paths used during the CUMULUS reproduction. Change the two path variables in the first configuration cell if necessary.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Open `CUMULUS_Reproduction.ipynb`.

Start with:

```python
RUN_MODE = "quick"
```

This evaluates all 27 pipelines on one file from each dataset at 5% corruption.

For the complete benchmark, use:

```python
RUN_MODE = "full"
```

The full run uses all recordings and all four corruption levels. Intermediate checkpoints are written locally to `results/checkpoints/` and are ignored by Git.


## Reproducibility note

The published paper specifies the method space, corruption models, stage order, and metric families. It does not fully specify every aggregation detail needed to reconstruct a unique scalar pipeline ranking. For transparency, the notebook keeps paper-reported reference tables separate from recomputed outputs and exports all intermediate metric vectors.
