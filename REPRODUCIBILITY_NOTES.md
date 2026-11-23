# AMQKE reproducibility notes

This note records implementation details observed in the supplied `AMQKE.ipynb` source. It does not change the manuscript or experimental results.

## 1. Preprocessing and potential data leakage

The experimental protocol describes preprocessing as being fitted on training data only. In the current code, however, `clean_numeric_features()` replaces missing-value markers, expands categorical variables, and calls `SimpleImputer(strategy="median").fit_transform(X)` while the full dataset is being loaded. This helper is used by the Heart Disease, Diabetes, Banknote, and Sonar loaders. The main evaluation later fits another imputer, scaler, and PCA on the training partition, but the earlier dataset-wide imputation has already used information from the eventual test partition.

For a strictly leakage-controlled rerun, modify the loaders to retain missing values (and handle categorical schema consistently), then fit all imputation, encoding, scaling, and dimensionality reduction within the training partition. Results may change after this correction; the existing tables should not be represented as results from the corrected pipeline unless experiments are rerun.

## 2. Manuscript data inputs versus default loader behavior

The code loads Breast Cancer from scikit-learn. It loads `heart.csv` locally when supplied, otherwise it attempts to fetch UCI dataset ID 45. Diabetes, Banknote, and Sonar are read from local `diabetes.csv`, `banknote.csv`, and `sonar.csv` files. These local data files are not part of this repository. The paper reports 289 Heart Disease observations, so supply the same local `heart.csv` used in the reported experiment if exact data-count reproduction is required; do not assume the fallback source yields identical samples or preprocessing history.

## 3. Kernel caching and outputs

The script stores generated kernel matrices in Python dictionaries for reuse during a single run. It does not persist kernels to disk, despite the manuscript's implementation/reproducibility description referring to caching. Similarly, the code prints numeric result tables and saves figures, but does not export result tables to CSV or JSON in this source snapshot.

## 4. Qiskit feature-map API

The source imports `ZFeatureMap`, `ZZFeatureMap`, and `PauliFeatureMap`. IBM's Qiskit API documents these feature-map classes as deprecated starting in Qiskit 2.1 and schedules their removal in Qiskit 3.0, recommending corresponding factory functions. This release pins Qiskit below 2.1 to preserve compatibility with the source as supplied. A migration should be tested before changing the dependency range or claiming compatibility with Qiskit 2.1+.

## 5. Version-label inconsistency

The notebook file is named `AMQKE.ipynb`, the source docstring identifies the implementation as `AMQKE-v25`, and the same docstring contains the line "Version 26: source/target average-alignment feature definition is unified." Confirm the intended version label before making a formal release or referencing a specific code version in the paper.

## 6. Execution verification

The Python export passes syntax compilation. A full experiment execution was not performed while preparing this repository because Qiskit, Qiskit Aer, and `ucimlrepo` were not installed in the available environment. The default run evaluates five datasets, five seeds, three noise levels, and three meta-learners, and is computationally intensive.
