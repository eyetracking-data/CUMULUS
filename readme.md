# CUMULUS

**CUMULUS** is a controlled benchmarking framework for evaluating preprocessing choices for eye-tracking time series. It evaluates preprocessing both at the level of individual stages and at the level of complete pipelines.

The repository accompanies the paper **“Robust Preprocessing Pipelines for Eye-Tracking Data”** by Jennifer Landes and Meike Klettke.

## Overview

Eye-tracking recordings frequently contain missing samples, tracking losses, and spike-like artifacts. CUMULUS evaluates preprocessing under controlled corruptions with known reference values instead of relying only on downstream classification performance.

The evaluated preprocessing stages are:

| Stage | Methods |
|---|---|
| Missing-value imputation | Mean, LOCF, KNN (`k = 5`) |
| Outlier handling | Z-score filtering, MAD filtering, Isolation Forest |
| Normalization | Min-Max scaling, Z-score standardization, Robust scaling |

Controlled missingness and spike corruptions are evaluated at **5%, 10%, 15%, and 20%**. The canonical signal representation contains four eye-tracking features:

```text
Gaze X
Gaze Y
ET_PupilLeft
ET_PupilRight
```

## Repository structure

```text
CUMULUS/
├── CUMULUS_Git_Stage_Evaluation/
│   ├── CUMULUS_Stage_Level_Evaluation.ipynb
│   ├── figures/
│   │   ├── figure_1_pipeline_design_space.png
│   │   └── figure_2_cumulus_workflow.png
│   └── results/
│       ├── CUMULUS_Output_Cheating.xlsx
│       └── CUMULUS_Output_d2.xlsx
├── CUMULUS_Pipeline_Level_Evaluation.ipynb
├── CUMULUS_pipeline_ranking.csv
└── README.md
```

## 1. Stage-level evaluation

[`CUMULUS_Stage_Level_Evaluation.ipynb`](CUMULUS_Git_Stage_Evaluation/CUMULUS_Stage_Level_Evaluation.ipynb) reproduces the stage-level benchmark.

It:

- maps heterogeneous eye-tracking exports to the common four-dimensional signal representation,
- injects controlled MCAR missingness and spike artifacts using fixed random seeds,
- evaluates imputation using RMSE and MAE,
- evaluates outlier handling using distributional deviation,
- evaluates normalization using KS statistics and distributional properties,
- writes the detailed and aggregated results to one Excel workbook per dataset.

The committed benchmark outputs are available in [`CUMULUS_Git_Stage_Evaluation/results/`](CUMULUS_Git_Stage_Evaluation/results/).

### Raw-data configuration

The raw eye-tracking datasets are not included in this repository. To rerun the stage-level benchmark from the original recordings, set the following environment variables to the corresponding local dataset directories:

```text
CUMULUS_D2_DIR
CUMULUS_CHEATING_DIR
```

The notebook supports CSV/TSV eye-tracking exports and creates the `results/` directory automatically.

### Main Python dependencies

```text
numpy
pandas
scipy
scikit-learn
jupyter
```

## 2. Pipeline-level evaluation

[`CUMULUS_Pipeline_Level_Evaluation.ipynb`](CUMULUS_Pipeline_Level_Evaluation.ipynb) reconstructs the ranking of complete preprocessing pipelines from the two stage-evaluation workbooks.

For every complete pipeline configuration:

1. the KS statistic is averaged across the four signal features within each dataset;
2. the Cheating and D2 dataset scores are averaged with equal weight;
3. pipelines are ranked by the resulting overall KS value.

Lower KS values indicate stronger preservation of the reference distribution.

The complete ranking is exported to:

[`CUMULUS_pipeline_ranking.csv`](CUMULUS_pipeline_ranking.csv)

The best-performing complete pipeline in this evaluation is:

```text
KNN imputation → Isolation Forest → Min-Max scaling
```

with an overall mean KS statistic of approximately **0.012449** across the two datasets.

The pipeline-level result complements the stage-level findings: a method that performs best in isolation does not necessarily form the strongest complete preprocessing pipeline.

## Reproducing the pipeline ranking

The two required workbooks are already included in:

```text
CUMULUS_Git_Stage_Evaluation/results/
```

The pipeline-level notebook reads:

```text
CUMULUS_Output_Cheating.xlsx
CUMULUS_Output_d2.xlsx
```

If the notebook is executed without first rerunning the stage-level benchmark, place these two workbooks in the notebook's current working directory or adapt the input paths accordingly.

## Evaluation outputs

The Excel workbooks contain the detailed benchmark results and aggregated summaries used by the notebooks. The executed notebooks retain their result tables so that the reported values can also be inspected directly on GitHub.

## Authors

**Jennifer Landes**  
**Meike Klettke**  
University of Regensburg, Faculty of Informatics and Data Science  
Chair of Data Engineering

## Citation

A formal citation will be added when the accompanying paper is published.
