# Cell Type Classification — Single-Cell RNA-Seq

**Course:** CS 412: Machine Learning | University of Illinois Chicago | Fall 2025  
**Author:** Bhavisha Ahuja

---

## Overview

Built a supervised ML classifier to identify 8 cell types from single-cell RNA sequencing (scRNA-seq) data. The dataset contains 20,000 labeled training cells and 10,000 unlabeled test cells, each described by expression levels across \~3,000 genes.

The core challenge: extreme class imbalance (two cell types make up \~85% of the data), high dimensionality, and extreme sparsity (most gene counts are zero due to dropout in sequencing).

---

## Problem & Approach

| Challenge | Approach |
| :---- | :---- |
| Class imbalance | `class_weight="balanced"` in Random Forest |
| High dimensionality (\~3,000 features) | RF handles natively; PCA explored but didn't improve performance |
| Sparse data | RF chosen over distance-based methods (no scaling needed) |
| Overfitting risk | Hyperparameter tuning: limited tree depth, increased min\_samples\_split |

---

## Models Compared

| Model | Validation Accuracy |
| :---- | :---- |
| Baseline Random Forest | 98.47% |
| Logistic Regression | \~94.7–98% |
| **Tuned Random Forest (final)** | **99.1%+** |

Logistic Regression underperformed on minority classes due to linear decision boundaries — scRNA-seq data has nonlinear structure. PCA reduced dimensionality but did not improve accuracy over the full-feature RF.

---

## Hyperparameter Tuning

Used 3-fold stratified cross-validation to tune:

- `n_estimators`: \[100, 200\] → best: **200**  
- `max_depth`: \[None, 20, 40\] → best: **20**  
- `min_samples_split`: \[2, 5\] → best: **5**

Limiting tree depth and increasing min\_samples\_split reduced overfitting on rare cell types (as few as 149 samples in one class).

---

## Results

The tuned Random Forest achieved **99.1%+ validation accuracy**, with improved recall on minority classes (mesothelial, ciliated epithelial). The final model was retrained on all 20,000 labeled samples before generating predictions for the 10,000 test cells.

---

## Files

| File | Description |
| :---- | :---- |
| `CS412_FinalProject.ipynb` | Main notebook with full pipeline and visualizations |
| `CS_412_finalProjectPyfile.py` | Python script version |
| `test_output.py` | Validation script for prediction output format |
| `Ahuja_Bhavisha_653160655.csv` | Final predictions (cell\_id, predicted\_cell\_type) |
| `CS 412 Final Project Report.pdf` | Full written report with analysis |

**Note:** Raw data files (cells\_batch1\_20000.csv, cells\_batch2\_10000.csv) are not included as they are course-provided datasets.

---

## Tech Stack

Python · scikit-learn · pandas · NumPy · matplotlib · seaborn · Jupyter Notebook  
