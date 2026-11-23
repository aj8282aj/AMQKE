# Adaptive Multi-Quantum Kernel Ensemble (AMQKE)

**Structural Diversity, Contribution-Aware Meta-Learning, and Dynamic Kernel Selection for Quantum Classification**

This repository contains the original Jupyter Notebook and its standalone Python implementation, together with setup, citation, licence, and reproducibility documentation. 

## Contents

- `AMQKE.ipynb` — supplied Jupyter Notebook.
- `AMQKE.py` — Python script exported from the notebook code cell.
- `requirements.txt` — pinned dependency ranges.
- `CITATION.cff` — software citation metadata.
- `LICENSE` — MIT License.
- `REPRODUCIBILITY_NOTES.md` — code–paper differences and limitations.
- `RELEASE_CHECKLIST.md` — pre-publication checks.

## Method overview

AMQKE constructs three fidelity-based quantum kernels from Qiskit feature maps: Z, ZZ, and PauliXY (a Pauli feature map configured with `paulis=["X", "YY"]` and linear entanglement). The default configuration uses four qubits and two feature-map repetitions.

The method measures pairwise kernel alignment and complementarity; forms per-kernel meta-features from cross-validated F1, accuracy, average complementarity, and average alignment; and uses cross-dataset Ridge Regression, Random Forest, and XGBoost regressors to predict leave-one-kernel-out F1 contributions. Predicted scores are converted to normalized weights. A separate dynamic-selection variant retains kernels with positive predicted contribution and falls back to the highest-scoring kernel if none is positive. Uniform fusion and adaptive-complementarity weighting are also evaluated.

Controlled depolarizing-noise conditions in the code are ideal (`p=0`), medium (`p=0.005`), and very high (`p=0.020`). These are simulator experiments, not physical QPU execution.

## Datasets and input files

The experiment uses five binary-classification datasets: Breast Cancer Wisconsin Diagnostic, Heart Disease, Diabetes, Banknote Authentication, and Sonar.

Breast Cancer is loaded from scikit-learn. Heart Disease is loaded from local `heart.csv` when available, otherwise the code attempts UCI dataset ID 45 via `ucimlrepo`. Diabetes, Banknote, and Sonar are expected as local `diabetes.csv`, `banknote.csv`, and `sonar.csv` files under `data/`. These CSV files are not included. Use the same source files and variants used for the reported manuscript results; see `REPRODUCIBILITY_NOTES.md` and check dataset redistribution terms.

The default protocol samples at most 160 observations, uses an 80/20 stratified split, caps training at 120 observations, and evaluates five seeds (`42, 52, 62, 72, 82`). For full details and configurable options, inspect `AMQKE.py --help`.

## Installation

Use Python 3.11 or later and create a virtual environment:

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current source uses Qiskit feature-map classes that are deprecated from Qiskit 2.1 and scheduled for removal in Qiskit 3.0. The requirements file therefore pins Qiskit below 2.1 for this code snapshot.

## Run

Place the required local CSV inputs under `data/`, then run:

```bash
python AMQKE.py --data-dir data --output-dir results
```

For a small smoke run:

```bash
python AMQKE.py --data-dir data --datasets breast_cancer,heart --seeds 42 --noise ideal --meta-learners ridge,random_forest --skip-plots
```

Use `python AMQKE.py --help` to see all command-line options. The notebook contains the same code in a single code cell and uses its default execution configuration.

## Outputs

The script prints per-run metrics, summary statistics, selection frequencies, and meta-learning diagnostics. It saves figures as PNG files. This source snapshot does not persist the kernel cache between processes or export the reported numeric result tables to CSV/JSON.

## Important reproducibility caveat

The experimental protocol specifies that preprocessing is fitted on the training partition only. However, in the supplied implementation, `clean_numeric_features()` performs median imputation during dataset loading, before the outer train/test split, for the Heart Disease, Diabetes, Banknote, and Sonar loaders. This means information from the eventual test partition can influence those imputation medians. A later train-split imputer/scaler/PCA does not reverse that earlier use of full-dataset information. See `REPRODUCIBILITY_NOTES.md`; correct the pipeline and rerun affected experiments before claiming a strictly leakage-free evaluation.

The script passed syntax compilation during package preparation; a complete Qiskit/Aer experiment run was not performed in that environment.

## Results interpretation

The manuscript reports dataset-dependent behavior and states that the classical RBF SVM outperformed the tested quantum-kernel configurations across the five datasets. AMQKE should therefore be described as a framework for adaptive analysis and integration of heterogeneous simulated quantum-kernel representations, not as evidence of universal quantum advantage.

## Citation and license

See `CITATION.cff` for citation metadata and `LICENSE` for the MIT License. The license covers original repository code and documentation only; datasets, Qiskit, and other third-party components remain under their respective terms. Confirm author, university, and funder permissions before treating the MIT release as final.
