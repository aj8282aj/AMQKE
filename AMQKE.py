#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMQKE-v25 — Cross-Dataset Contribution-Aware Meta-Learning
============================================================
Research-aligned implementation for the AMQKE manuscript.
Version 26: source/target average-alignment feature definition is unified.

Core design
-----------
1) Z, ZZ and PauliXY quantum feature maps.
2) Fidelity quantum kernels (pure-state fidelity for ideal simulation and
   Uhlmann/state fidelity for noisy density matrices).
3) Centered kernel alignment and structural complementarity.
4) Source-dataset meta-learning: the held-out target dataset is NEVER used to
   fit the meta-learner.
5) Meta-learners:
     - Ridge Regression
     - Random Forest Regressor
     - XGBoost Regressor (optional dependency)
6) Meta target = empirical marginal kernel contribution on source datasets,
   estimated entirely from source training folds:
       Delta_i = F1(all-kernel fusion) - F1(fusion without kernel i)
7) Target meta-features are observable from target training data only:
       [CV_F1_i, CV_Accuracy_i, Average_Complementarity_i,
        Average_Alignment_i]
   The target marginal contribution is deliberately NOT used as an input,
   avoiding circular prediction.
8) Predicted contribution scores are converted to non-negative adaptive
   weights, with an additional contribution-based dynamic selection mode.
9) Uniform fusion and adaptive-complementarity weighting are retained as
   non-meta baselines.
10) Controlled depolarizing noise: ideal / p=0.005 / p=0.020.

The script is intentionally conservative about claims: cross-dataset
meta-learning is implemented as a transfer-of-kernel-contribution mechanism,
not as evidence of quantum advantage.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from qiskit import transpile
    from qiskit.circuit.library import ZFeatureMap, ZZFeatureMap, PauliFeatureMap
    from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Qiskit/Qiskit Aer are required. Install with:\n"
        "  pip install qiskit qiskit-aer\n\n"
        f"Import error: {exc}"
    )

warnings.filterwarnings("ignore", category=FutureWarning)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FEATURE_MAP_NAMES = ("Z", "ZZ", "PauliXY")
NOISE_LEVELS = {"ideal": 0.000, "medium": 0.005, "very_high": 0.020}
DEFAULT_SEEDS = (42, 52, 62, 72, 82)
DEFAULT_MAX_TRAIN_SAMPLES = 120
DEFAULT_MAX_TEST_SAMPLES = 40
DEFAULT_TEST_SIZE = 0.20
DEFAULT_QUBITS = 4
DEFAULT_REPS = 2
DEFAULT_BATCH_SIZE = 16
DEFAULT_N_META_SPLITS = 3
DEFAULT_META_LEARNERS = ("ridge", "random_forest", "xgboost")
DEFAULT_WEIGHT_METHOD = "all"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    display_name: str
    local_filename: str | None = None


DATASETS = {
    "breast_cancer": DatasetSpec("breast_cancer", "Breast Cancer Wisconsin Diagnostic"),
    "heart": DatasetSpec("heart", "Heart Disease", "heart.csv"),
    "diabetes": DatasetSpec("diabetes", "Pima Indians Diabetes", "diabetes.csv"),
    "banknote": DatasetSpec("banknote", "Banknote Authentication", "banknote.csv"),
    "sonar": DatasetSpec("sonar", "Sonar", "sonar.csv"),
}


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def softmax(values: np.ndarray, temperature: float = 0.50) -> np.ndarray:
    x = np.asarray(values, dtype=float) / max(float(temperature), 1e-8)
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(np.sum(e), 1e-12)


def normalize_01(values: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = float(np.min(x)), float(np.max(x))
    if math.isclose(lo, hi):
        return np.ones_like(x) * 0.5
    return (x - lo) / (hi - lo)


def metric_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def ensure_binary_target(y: Iterable) -> np.ndarray:
    arr = pd.Series(y).replace({"?": np.nan}).dropna()
    if not pd.api.types.is_numeric_dtype(arr):
        codes, _ = pd.factorize(arr.astype(str))
        if len(np.unique(codes)) != 2:
            raise ValueError("Target is not binary.")
        return codes.astype(int)
    values = arr.astype(float).to_numpy()
    unique = np.unique(values)
    if len(unique) == 2:
        mapping = {unique[0]: 0, unique[1]: 1}
        return np.array([mapping[v] for v in values], dtype=int)
    if 0 in unique:
        return (values > 0).astype(int)
    median = np.median(values)
    return (values > median).astype(int)


def clean_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy().replace(["?", "NA", "N/A", "null", "None", ""], np.nan)
    for c in X.columns:
        try:
            X[c] = pd.to_numeric(X[c])
        except Exception:
            pass
    X = pd.get_dummies(X, drop_first=False)
    X = X.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    return pd.DataFrame(imputer.fit_transform(X), columns=list(X.columns))


def take_stratified_sample(X: np.ndarray, y: np.ndarray, max_samples: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(X) <= max_samples:
        return X, y
    X_sub, _, y_sub, _ = train_test_split(
        X, y, train_size=max_samples, random_state=seed, stratify=y
    )
    return X_sub, y_sub


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_breast_cancer_dataset() -> Tuple[np.ndarray, np.ndarray]:
    ds = load_breast_cancer()
    return ds.data.astype(float), ds.target.astype(int)


def load_heart_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    local = data_dir / "heart.csv"
    if local.exists():
        df = pd.read_csv(local)
        target = next((c for c in ["output", "Output", "target", "Target", "condition", "num", "label"] if c in df.columns), df.columns[-1])
        y = ensure_binary_target(df[target])
        X = clean_numeric_features(df.drop(columns=[target]))
        return X.to_numpy(float), y
    try:
        from ucimlrepo import fetch_ucirepo
    except Exception as exc:
        raise RuntimeError("Install ucimlrepo or provide heart.csv") from exc
    heart = fetch_ucirepo(id=45)
    X = clean_numeric_features(heart.data.features)
    y = ensure_binary_target(heart.data.targets.iloc[:, 0])
    n = min(len(X), len(y))
    return X.iloc[:n].to_numpy(float), y[:n]


def load_last_column_csv(path: Path, sonar: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    df0 = pd.read_csv(path)
    try:
        float(str(df0.columns[-1]))
        df = pd.read_csv(path, header=None)
    except Exception:
        df = df0
    y = ensure_binary_target(df.iloc[:, -1]) if sonar else ensure_binary_target(df.iloc[:, -1])
    X = clean_numeric_features(df.iloc[:, :-1])
    return X.to_numpy(float), y


def load_diabetes_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    path = data_dir / "diabetes.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    target = next((c for c in ["Outcome", "outcome", "Class", "class", "target", "Target"] if c in df.columns), df.columns[-1])
    y = ensure_binary_target(df[target])
    X = clean_numeric_features(df.drop(columns=[target]))
    return X.to_numpy(float), y


def load_dataset(name: str, data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    if name == "breast_cancer":
        return load_breast_cancer_dataset()
    if name == "heart":
        return load_heart_dataset(data_dir)
    if name == "diabetes":
        return load_diabetes_dataset(data_dir)
    if name == "banknote":
        return load_last_column_csv(data_dir / "banknote.csv")
    if name == "sonar":
        return load_last_column_csv(data_dir / "sonar.csv", sonar=True)
    raise KeyError(name)


# ---------------------------------------------------------------------------
# Quantum feature maps / kernels
# ---------------------------------------------------------------------------


def build_feature_maps(num_qubits: int, reps: int) -> Dict[str, object]:
    return {
        "Z": ZFeatureMap(feature_dimension=num_qubits, reps=reps),
        "ZZ": ZZFeatureMap(feature_dimension=num_qubits, reps=reps, entanglement="linear"),
        "PauliXY": PauliFeatureMap(
            feature_dimension=num_qubits,
            reps=reps,
            paulis=["X", "YY"],
            entanglement="linear",
        ),
    }


def angle_encode(X: np.ndarray, num_qubits: int) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.shape[1] < num_qubits:
        X = np.hstack([X, np.zeros((len(X), num_qubits - X.shape[1]))])
    X = X[:, :num_qubits]
    return np.tanh(X) * np.pi


def bind_feature_map(feature_map, x: np.ndarray):
    params = list(feature_map.parameters)
    if len(params) != len(x):
        raise ValueError(f"Feature map expects {len(params)} parameters but sample has {len(x)} values")
    return feature_map.assign_parameters(dict(zip(params, [float(v) for v in x])), inplace=False)


def make_depolarizing_noise_model(p: float) -> NoiseModel | None:
    if p <= 0:
        return None
    noise = NoiseModel()
    one = depolarizing_error(p, 1)
    two = depolarizing_error(p, 2)
    noise.add_all_qubit_quantum_error(one, ["rz", "sx", "x", "h"])
    noise.add_all_qubit_quantum_error(two, ["cx"])
    return noise


def generate_states(X: np.ndarray, feature_map, p: float, seed: int, batch_size: int = 16) -> List[object]:
    circuits = [bind_feature_map(feature_map, row) for row in X]
    if p <= 0:
        return [Statevector.from_instruction(c) for c in circuits]
    simulator = AerSimulator(
        method="density_matrix",
        noise_model=make_depolarizing_noise_model(p),
        seed_simulator=seed,
    )
    states: List[object] = []
    for start in range(0, len(circuits), batch_size):
        batch = circuits[start : start + batch_size]
        transpiled = transpile(batch, simulator, basis_gates=["rz", "sx", "x", "cx"], optimization_level=0)
        run_circuits = []
        for c in transpiled:
            c2 = c.copy()
            c2.save_density_matrix()
            run_circuits.append(c2)
        result = simulator.run(run_circuits, shots=1, seed_simulator=seed + start).result()
        for i in range(len(run_circuits)):
            data = result.data(i)
            dm = None
            try:
                dm = data["density_matrix"]
            except Exception:
                pass
            if dm is None:
                try:
                    dm = data.get("density_matrix")
                except Exception:
                    pass
            if dm is None:
                dm = getattr(data, "density_matrix", None)
            if dm is None:
                raise RuntimeError(f"Density matrix extraction failed; available keys={list(data.keys()) if hasattr(data, 'keys') else 'unknown'}")
            states.append(DensityMatrix(dm))
    return states


def fidelity_kernel_from_states(states_a: Sequence[object], states_b: Sequence[object] | None = None) -> np.ndarray:
    same = states_b is None
    if states_b is None:
        states_b = states_a
    K = np.zeros((len(states_a), len(states_b)), dtype=float)
    for i, a in enumerate(states_a):
        for j, b in enumerate(states_b):
            if same and j < i:
                K[i, j] = K[j, i]
                continue
            val = float(np.real(state_fidelity(a, b)))
            K[i, j] = np.clip(val, 0.0, 1.0)
    if same:
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 1.0)
    return K


def prepare_kernel_set(X: np.ndarray, maps: Dict[str, object], p: float, seed: int, batch_size: int) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in FEATURE_MAP_NAMES:
        print(f"      generating {name} | noise={p:.3f}")
        states = generate_states(X, maps[name], p, seed, batch_size=batch_size)
        out[name] = fidelity_kernel_from_states(states)
    return out


# ---------------------------------------------------------------------------
# Kernel structure / SVM
# ---------------------------------------------------------------------------


def center_kernel(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def centered_kernel_alignment(K: np.ndarray, L: np.ndarray) -> float:
    Kc, Lc = center_kernel(K), center_kernel(L)
    denom = np.linalg.norm(Kc, "fro") * np.linalg.norm(Lc, "fro")
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(Kc * Lc) / denom)


def alignment_and_complementarity(kernels: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = len(kernels)
    A = np.eye(m, dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            v = centered_kernel_alignment(kernels[i], kernels[j])
            A[i, j] = A[j, i] = v
    C = 1.0 - A
    np.fill_diagonal(C, 0.0)
    avg_comp = np.sum(C, axis=1) / max(m - 1, 1)
    return A, C, avg_comp


def precomputed_svc_fit_predict(K_train, K_test_train, y_train, y_test) -> Dict[str, float]:
    clf = SVC(kernel="precomputed", C=1.0)
    clf.fit(K_train, y_train)
    pred = clf.predict(K_test_train)
    return metric_dict(y_test, pred)


def subset_kernel(K: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray | None = None) -> np.ndarray:
    if idx_b is None:
        idx_b = idx_a
    return K[np.ix_(idx_a, idx_b)]


def fused_kernel_full(kernels: Sequence[np.ndarray], weights: np.ndarray | None = None) -> np.ndarray:
    if weights is None:
        weights = np.ones(len(kernels), dtype=float) / len(kernels)
    K = np.zeros_like(kernels[0], dtype=float)
    for w, Ki in zip(weights, kernels):
        K += float(w) * Ki
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 1.0)
    return K


def cv_kernel_metrics(K: np.ndarray, y: np.ndarray, train_idx: np.ndarray, seed: int, n_splits: int = 3) -> Tuple[float, float]:
    idx = np.asarray(train_idx)
    n_splits = max(2, min(n_splits, int(np.min(np.bincount(y[idx])))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    f1s, accs = [], []
    for a, b in skf.split(idx, y[idx]):
        tr, va = idx[a], idx[b]
        m = precomputed_svc_fit_predict(subset_kernel(K, tr), subset_kernel(K, va, tr), y[tr], y[va])
        f1s.append(m["f1"])
        accs.append(m["accuracy"])
    return float(np.mean(f1s)), float(np.mean(accs))


def cv_fusion_f1(kernels: Sequence[np.ndarray], y: np.ndarray, train_idx: np.ndarray, seed: int, keep: Sequence[int]) -> float:
    idx = np.asarray(train_idx)
    keep = list(keep)
    selected = [kernels[i] for i in keep]
    n_splits = max(2, min(3, int(np.min(np.bincount(y[idx])))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 17)
    f1s = []
    uniform = np.ones(len(selected), dtype=float) / len(selected)
    Kf = fused_kernel_full(selected, uniform)
    for a, b in skf.split(idx, y[idx]):
        tr, va = idx[a], idx[b]
        m = precomputed_svc_fit_predict(subset_kernel(Kf, tr), subset_kernel(Kf, va, tr), y[tr], y[va])
        f1s.append(m["f1"])
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# Baseline adaptive-complementarity weighting
# ---------------------------------------------------------------------------


def adaptive_complementarity_weights(kernels: Sequence[np.ndarray], y: np.ndarray, train_idx: np.ndarray, seed: int):
    train_k = [subset_kernel(K, train_idx) for K in kernels]
    A, C, avg_comp = alignment_and_complementarity(train_k)
    perf = np.array([cv_kernel_metrics(K, y, train_idx, seed)[0] for K in kernels])
    raw = 0.70 * normalize_01(perf) + 0.30 * normalize_01(avg_comp)
    return softmax(raw, 0.50), perf, A, avg_comp


# ---------------------------------------------------------------------------
# Cross-dataset meta-learning
# ---------------------------------------------------------------------------


def source_meta_features_and_targets(
    kernels: Sequence[np.ndarray],
    y: np.ndarray,
    train_idx: np.ndarray,
    seed: int,
    n_splits: int = DEFAULT_N_META_SPLITS,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Create fold-level source meta samples.

    Each source fold generates one row per kernel.  This increases the effective
    meta-training sample count while keeping the contribution target strictly
    inside the source dataset/fold.

    Features:
      [validation_F1, validation_accuracy,
       fold_average_complementarity, fold_average_alignment]

    Target:
      Delta_i = F1(full uniform fusion) - F1(fusion without kernel i)
      measured on the same source validation fold.
    """
    idx = np.asarray(train_idx)
    y_local = y[idx]
    min_class = int(np.min(np.bincount(y_local)))
    n_splits = max(2, min(n_splits, min_class))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    X_rows: List[List[float]] = []
    y_rows: List[float] = []
    diagnostics = {
        "f1": [],
        "accuracy": [],
        "avg_complementarity": [],
        "avg_alignment": [],
        "contribution": [],
    }

    for fold_id, (a, b) in enumerate(skf.split(idx, y_local)):
        fold_train = idx[a]
        fold_valid = idx[b]

        fold_train_k = [subset_kernel(K, fold_train) for K in kernels]
        A, C, avg_comp = alignment_and_complementarity(fold_train_k)
        avg_align = np.sum(A - np.eye(len(kernels)), axis=1) / max(len(kernels) - 1, 1)

        val_f1 = []
        val_acc = []
        val_train_blocks = []
        val_test_blocks = []

        for K in kernels:
            K_fit = subset_kernel(K, fold_train)
            K_val = subset_kernel(K, fold_valid, fold_train)
            m = precomputed_svc_fit_predict(
                K_fit, K_val, y[fold_train], y[fold_valid]
            )
            val_f1.append(m["f1"])
            val_acc.append(m["accuracy"])

        # Uniform full ensemble on the validation fold.
        uniform = np.ones(len(kernels), dtype=float) / len(kernels)
        K_full_train = fused_kernel_full(fold_train_k, uniform)
        K_full_valid = np.zeros((len(fold_valid), len(fold_train)), dtype=float)
        for w, K in zip(uniform, kernels):
            K_full_valid += float(w) * subset_kernel(K, fold_valid, fold_train)
        full_metrics = precomputed_svc_fit_predict(
            K_full_train, K_full_valid, y[fold_train], y[fold_valid]
        )
        full_f1 = full_metrics["f1"]

        fold_deltas = []
        for i in range(len(kernels)):
            keep = [j for j in range(len(kernels)) if j != i]
            w_loo = np.ones(len(keep), dtype=float) / len(keep)
            loo_train = fused_kernel_full(
                [fold_train_k[j] for j in keep], w_loo
            )
            loo_valid = np.zeros((len(fold_valid), len(fold_train)), dtype=float)
            for w, j in zip(w_loo, keep):
                loo_valid += float(w) * subset_kernel(kernels[j], fold_valid, fold_train)
            loo_metrics = precomputed_svc_fit_predict(
                loo_train, loo_valid, y[fold_train], y[fold_valid]
            )
            fold_deltas.append(full_f1 - loo_metrics["f1"])

        for i in range(len(kernels)):
            X_rows.append([
                float(val_f1[i]),
                float(val_acc[i]),
                float(avg_comp[i]),
                float(avg_align[i]),
            ])
            y_rows.append(float(fold_deltas[i]))
            diagnostics["f1"].append(float(val_f1[i]))
            diagnostics["accuracy"].append(float(val_acc[i]))
            diagnostics["avg_complementarity"].append(float(avg_comp[i]))
            diagnostics["avg_alignment"].append(float(avg_align[i]))
            diagnostics["contribution"].append(float(fold_deltas[i]))

    return np.asarray(X_rows, dtype=float), np.asarray(y_rows, dtype=float), diagnostics


def make_meta_estimator(name: str, seed: int):
    if name == "ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=2,
            random_state=seed,
        )
    if name == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not installed. Install it with: pip install xgboost")
        return XGBRegressor(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=1,
        )
    raise ValueError(f"Unknown meta learner: {name}")


def target_meta_features(
    kernels: Sequence[np.ndarray], y: np.ndarray, train_idx: np.ndarray, seed: int
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    train_k = [subset_kernel(K, train_idx) for K in kernels]
    A, C, avg_comp = alignment_and_complementarity(train_k)
    perf, acc = [], []
    for K in kernels:
        f1, accuracy = cv_kernel_metrics(K, y, train_idx, seed + 11)
        perf.append(f1)
        acc.append(accuracy)
    perf = np.asarray(perf, dtype=float)
    acc = np.asarray(acc, dtype=float)
    avg_align = np.sum(A - np.eye(len(kernels)), axis=1) / max(len(kernels) - 1, 1)
    X_target = np.column_stack([perf, acc, avg_comp, avg_align])
    diag = {"f1": perf, "accuracy": acc, "alignment": A, "avg_complementarity": avg_comp, "avg_alignment": avg_align}
    return X_target, diag


def cross_dataset_meta_weights(
    target_name: str,
    dataset_cache: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]],
    train_indices: Dict[str, np.ndarray],
    seed: int,
    meta_name: str,
    n_splits: int = DEFAULT_N_META_SPLITS,
) -> Dict[str, object]:
    X_rows: List[np.ndarray] = []
    y_rows: List[np.ndarray] = []
    source_names = []
    source_diags = []

    for source_name, (kernel_dict, y_source, _) in dataset_cache.items():
        if source_name == target_name:
            continue
        kernels = [kernel_dict[n] for n in FEATURE_MAP_NAMES]
        X_m, y_m, diag = source_meta_features_and_targets(
            kernels, y_source, train_indices[source_name], seed, n_splits=n_splits
        )
        X_rows.append(X_m)
        y_rows.append(y_m)
        source_names.append(source_name)
        source_diags.append(diag)

    if not X_rows:
        raise RuntimeError("No source datasets are available for cross-dataset meta-learning.")

    X_train_meta = np.vstack(X_rows)
    y_train_meta = np.concatenate(y_rows)
    model = make_meta_estimator(meta_name, seed)
    model.fit(X_train_meta, y_train_meta)

    target_kernel_dict, target_y, _ = dataset_cache[target_name]
    target_kernels = [target_kernel_dict[n] for n in FEATURE_MAP_NAMES]
    X_target, target_diag = target_meta_features(target_kernels, target_y, train_indices[target_name], seed)
    predicted = np.asarray(model.predict(X_target), dtype=float)

    weights_all = softmax(predicted, temperature=0.50)
    selected = np.flatnonzero(predicted > 0.0)
    if len(selected) == 0:
        selected = np.array([int(np.argmax(predicted))], dtype=int)
    weights_selected = np.zeros(len(FEATURE_MAP_NAMES), dtype=float)
    weights_selected[selected] = softmax(predicted[selected], temperature=0.50)

    return {
        "weights_all": weights_all,
        "weights_selected": weights_selected,
        "predicted_contribution": predicted,
        "selected_indices": selected,
        "target_meta_features": X_target,
        "target_diag": target_diag,
        "n_source_meta_rows": int(len(y_train_meta)),
        "source_datasets": source_names,
        "meta_model": model,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_target_seed(
    target_name: str,
    dataset_cache: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]],
    train_indices: Dict[str, np.ndarray],
    test_indices: Dict[str, np.ndarray],
    seed: int,
    meta_learners: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    kernels_dict, y, _ = dataset_cache[target_name]
    train_idx = train_indices[target_name]
    test_idx = test_indices[target_name]
    kernels = [kernels_dict[n] for n in FEATURE_MAP_NAMES]
    rows = []

    # Individual kernels.
    individual_f1 = []
    for name, K in zip(FEATURE_MAP_NAMES, kernels):
        m = precomputed_svc_fit_predict(subset_kernel(K, train_idx), subset_kernel(K, test_idx, train_idx), y[train_idx], y[test_idx])
        individual_f1.append(m["f1"])
        rows.append({"model": f"{name} Kernel", "selection": "none", **m, **{f"weight_{n}": np.nan for n in FEATURE_MAP_NAMES}})

    # Uniform fusion.
    uniform = np.ones(len(kernels), dtype=float) / len(kernels)
    Ku = fused_kernel_full(kernels, uniform)
    um = precomputed_svc_fit_predict(subset_kernel(Ku, train_idx), subset_kernel(Ku, test_idx, train_idx), y[train_idx], y[test_idx])
    rows.append({"model": "Uniform Quantum Ensemble", "selection": "none", **um, **{f"weight_{n}": 1.0 / len(kernels) for n in FEATURE_MAP_NAMES}})

    # Legacy adaptive-complementarity baseline.
    ac_w, ac_perf, ac_A, ac_comp = adaptive_complementarity_weights(kernels, y, train_idx, seed)
    Kac = fused_kernel_full(kernels, ac_w)
    acm = precomputed_svc_fit_predict(subset_kernel(Kac, train_idx), subset_kernel(Kac, test_idx, train_idx), y[train_idx], y[test_idx])
    rows.append({"model": "Adaptive Complementarity", "selection": "none", **acm, **{f"weight_{n}": float(ac_w[i]) for i, n in enumerate(FEATURE_MAP_NAMES)}})

    meta_records = {}
    for meta_name in meta_learners:
        info = cross_dataset_meta_weights(
            target_name, dataset_cache, train_indices, seed, meta_name
        )
        meta_records[meta_name] = info

        # Continuous contribution-aware fusion.
        Ka = fused_kernel_full(kernels, info["weights_all"])
        am = precomputed_svc_fit_predict(subset_kernel(Ka, train_idx), subset_kernel(Ka, test_idx, train_idx), y[train_idx], y[test_idx])
        rows.append({
            "model": f"Contribution-Aware {meta_name}",
            "selection": "none",
            **am,
            **{f"weight_{n}": float(info["weights_all"][i]) for i, n in enumerate(FEATURE_MAP_NAMES)},
            **{f"pred_contribution_{n}": float(info["predicted_contribution"][i]) for i, n in enumerate(FEATURE_MAP_NAMES)},
        })

        # Dynamic contribution-based selection.
        Ks = fused_kernel_full(kernels, info["weights_selected"])
        sm = precomputed_svc_fit_predict(subset_kernel(Ks, train_idx), subset_kernel(Ks, test_idx, train_idx), y[train_idx], y[test_idx])
        rows.append({
            "model": f"Contribution+Selection {meta_name}",
            "selection": ",".join(FEATURE_MAP_NAMES[i] for i in info["selected_indices"]),
            **sm,
            **{f"weight_{n}": float(info["weights_selected"][i]) for i, n in enumerate(FEATURE_MAP_NAMES)},
            **{f"pred_contribution_{n}": float(info["predicted_contribution"][i]) for i, n in enumerate(FEATURE_MAP_NAMES)},
        })

    # Structural diagnostics from target training data.
    A, C, avg_comp = alignment_and_complementarity([subset_kernel(K, train_idx) for K in kernels])
    selected_for_summary = meta_records[meta_learners[0]]["selected_indices"] if meta_learners else np.array([int(np.argmax(ac_w))])
    summary = {
        "best_individual_f1": float(np.max(individual_f1)),
        "uniform_f1": float(um["f1"]),
        "adaptive_complementarity_f1": float(acm["f1"]),
        "adaptive_complementarity_gain": float(acm["f1"] - np.max(individual_f1)),
        "mean_pairwise_complementarity": float(np.mean(C[np.triu_indices(len(kernels), 1)])),
        "mean_average_complementarity": float(np.mean(avg_comp)),
        "selected_primary_meta": ",".join(FEATURE_MAP_NAMES[i] for i in selected_for_summary),
        "alignment_matrix": A,
        "complementarity_matrix": C,
        "meta_records": meta_records,
    }
    return pd.DataFrame(rows), summary


# ---------------------------------------------------------------------------
# Classical baseline
# ---------------------------------------------------------------------------


def classical_rbf_metrics(X_raw: np.ndarray, y: np.ndarray, seed: int, test_size: float) -> Dict[str, float]:
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)
    clf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("svc", SVC(kernel="rbf", C=1.0))])
    clf.fit(X_raw[tr], y[tr])
    return metric_dict(y[te], clf.predict(X_raw[te]))


# ---------------------------------------------------------------------------
# Simple figures
# ---------------------------------------------------------------------------


def save_basic_figures(results: pd.DataFrame, out_dir: Path) -> None:
    """Save figures only. No CSV, TXT, JSON, or PDF outputs are written."""
    out_dir.mkdir(parents=True, exist_ok=True)

    ideal = results[results["noise"] == "ideal"].copy()
    if not ideal.empty:
        piv = ideal.groupby(["dataset", "model"], as_index=False)["f1"].mean()
        wide = piv.pivot(index="dataset", columns="model", values="f1")
        ax = wide.plot(kind="bar", figsize=(12, 6))
        ax.set_ylabel("Mean F1-score")
        ax.set_xlabel("")
        ax.set_title("AMQKE-v25 Performance Comparison Under Ideal Simulation")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "Figure_1_Performance_Comparison.png", dpi=250)
        plt.close()

    if "mean_pairwise_complementarity" in results.columns:
        p = results.groupby(["dataset", "noise"], as_index=False)["mean_pairwise_complementarity"].mean()
        if not p.empty:
            wide = p.pivot(index="dataset", columns="noise", values="mean_pairwise_complementarity")
            ax = wide.T.plot(marker="o", figsize=(9, 6))
            ax.set_xlabel("Noise condition")
            ax.set_ylabel("Mean structural diversity (1 - alignment)")
            ax.set_title("Structural Diversity vs. Depolarizing Noise")
            plt.tight_layout()
            plt.savefig(out_dir / "Figure_2_Diversity_vs_Noise.png", dpi=250)
            plt.close()

    # Figure 3: mean continuous meta-learned weights at ideal noise.
    ideal_adaptive = results[
        (results["noise"] == "ideal")
        & results["model"].str.startswith("Contribution-Aware")
        & (results["selection"] == "none")
    ].copy()
    if not ideal_adaptive.empty:
        cols = [f"weight_{n}" for n in FEATURE_MAP_NAMES]
        means = ideal_adaptive.groupby("model")[cols].mean()
        ax = means.plot(kind="bar", figsize=(10, 6))
        ax.set_ylabel("Mean adaptive weight")
        ax.set_xlabel("")
        ax.set_title("Cross-Dataset Meta-Learned Kernel Weights (Ideal)")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "Figure_3_MetaLearned_Weights.png", dpi=250)
        plt.close()

    # Figure 4: dynamic-selection frequency by target dataset and noise.
    sel = results[results["model"].str.startswith("Contribution+Selection")].copy()
    if not sel.empty:
        freq = (
            sel.groupby(["dataset", "noise", "model", "selection"], as_index=False)
               .size()
               .rename(columns={"size": "count"})
        )
        freq["total"] = freq.groupby(["dataset", "noise", "model"])["count"].transform("sum")
        freq["frequency_pct"] = 100.0 * freq["count"] / freq["total"]
        # Keep the most frequent selection per target/noise/model in the chart.
        top = freq.sort_values(
            ["dataset", "noise", "model", "frequency_pct"], ascending=[True, True, True, False]
        ).groupby(["dataset", "noise", "model"], as_index=False).head(1)
        pivot_key = top["model"].str.replace("Contribution+Selection ", "", regex=False)
        top = top.copy()
        top["learner"] = pivot_key
        top["label"] = top["dataset"] + " | " + top["noise"] + " | " + top["learner"]
        ax = top.sort_values("frequency_pct").plot(
            x="label", y="frequency_pct", kind="barh", figsize=(11, 8), legend=False
        )
        ax.set_xlabel("Selection frequency (%)")
        ax.set_ylabel("")
        ax.set_title("Most Frequent Dynamic Kernel Selection")
        plt.tight_layout()
        plt.savefig(out_dir / "Figure_4_Selection_Frequency_Noise.png", dpi=250)
        plt.close()

    # Figure 5: mean meta-learned weights across noise conditions.
    all_adaptive = results[
        (results["model"].str.startswith("Contribution-Aware"))
        & (results["selection"] == "none")
    ].copy()
    if not all_adaptive.empty:
        cols = [f"weight_{n}" for n in FEATURE_MAP_NAMES]
        g = all_adaptive.groupby("noise")[cols].mean()
        ax = g.plot(kind="bar", figsize=(9, 6))
        ax.set_ylabel("Mean meta-learned weight")
        ax.set_xlabel("Noise condition")
        ax.set_title("Meta-Learned Kernel Weights Across Noise Conditions")
        plt.tight_layout()
        plt.savefig(out_dir / "Figure_5_MetaWeights_vs_Noise.png", dpi=250)
        plt.close()


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AMQKE-v25 cross-dataset meta-learning experiment")
    p.add_argument("--data-dir", type=Path, default=Path("."))
    p.add_argument("--output-dir", type=Path, default=Path("amqke_v24_figures"))
    p.add_argument("--datasets", default=",".join(DATASETS.keys()))
    p.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    p.add_argument("--noise", default="ideal,medium,very_high")
    p.add_argument("--max-train-samples", type=int, default=DEFAULT_MAX_TRAIN_SAMPLES)
    p.add_argument("--max-test-samples", type=int, default=DEFAULT_MAX_TEST_SAMPLES)
    p.add_argument("--qubits", type=int, default=DEFAULT_QUBITS)
    p.add_argument("--reps", type=int, default=DEFAULT_REPS)
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--meta-learners", default=",".join(DEFAULT_META_LEARNERS))
    p.add_argument("--skip-plots", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names = [x.strip() for x in args.datasets.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    noise_names = [x.strip() for x in args.noise.split(",") if x.strip()]
    meta_learners = [x.strip().lower() for x in args.meta_learners.split(",") if x.strip()]

    for m in meta_learners:
        if m not in {"ridge", "random_forest", "xgboost"}:
            raise SystemExit(f"Unknown meta learner: {m}")
        if m == "xgboost" and XGBRegressor is None:
            raise SystemExit("xgboost requested but not installed. Run: pip install xgboost")

    unknown_datasets = [d for d in dataset_names if d not in DATASETS]
    if unknown_datasets:
        raise SystemExit(f"Unknown datasets: {unknown_datasets}")
    unknown_noise = [n for n in noise_names if n not in NOISE_LEVELS]
    if unknown_noise:
        raise SystemExit(f"Unknown noise levels: {unknown_noise}")
    if len(dataset_names) < 2:
        raise SystemExit("Cross-dataset meta-learning requires at least two datasets.")

    maps = build_feature_maps(args.qubits, args.reps)
    all_results: List[pd.DataFrame] = []
    all_summary: List[Dict[str, object]] = []

    # Precompute kernels once per dataset/noise/seed and retain them for the
    # cross-dataset meta-learning stage. This is essential: the target dataset
    # is excluded from meta-model fitting, but source kernels remain available.
    for seed in seeds:
        dataset_raw: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for dname in dataset_names:
            X_raw, y_raw = load_dataset(dname, args.data_dir)
            # Use the same deterministic seed for each dataset's bounded sample.
            X_raw, y_raw = take_stratified_sample(X_raw, y_raw, args.max_train_samples + args.max_test_samples, seed=seed)
            dataset_raw[dname] = (X_raw, y_raw)

        for noise_name in noise_names:
            p_noise = NOISE_LEVELS[noise_name]
            dataset_cache: Dict[str, Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]] = {}
            train_indices: Dict[str, np.ndarray] = {}
            test_indices: Dict[str, np.ndarray] = {}

            print(f"\n=== seed={seed} | noise={noise_name} (p={p_noise:.3f}) ===")

            for dname in dataset_names:
                X_raw, y = dataset_raw[dname]
                idx = np.arange(len(y))
                tr, te = train_test_split(idx, test_size=args.test_size, random_state=seed, stratify=y)
                if len(tr) > args.max_train_samples:
                    tr, _ = train_test_split(tr, train_size=args.max_train_samples, random_state=seed, stratify=y[tr])
                if len(te) > args.max_test_samples:
                    te, _ = train_test_split(te, train_size=args.max_test_samples, random_state=seed, stratify=y[te])
                train_indices[dname] = np.asarray(tr)
                test_indices[dname] = np.asarray(te)

                # Leakage-controlled preprocessing: fit imputer/scaler/PCA on
                # the training partition only, then transform the full sampled
                # matrix so that train-train and test-train kernel blocks use
                # exactly the same fitted preprocessing pipeline.
                imputer = SimpleImputer(strategy="median")
                X_imp_train = imputer.fit_transform(X_raw[tr])
                X_imp_all = imputer.transform(X_raw)
                scaler = StandardScaler()
                X_sc_train = scaler.fit_transform(X_imp_train)
                X_sc_all = scaler.transform(X_imp_all)

                n_pca = min(
                    args.qubits,
                    X_sc_train.shape[1],
                    max(1, X_sc_train.shape[0] - 1),
                )
                if n_pca >= 1 and X_sc_train.shape[1] > args.qubits:
                    pca = PCA(n_components=n_pca, random_state=seed)
                    X_train_processed = pca.fit_transform(X_sc_train)
                    X_all_processed = pca.transform(X_sc_all)
                else:
                    X_train_processed = X_sc_train
                    X_all_processed = X_sc_all

                Xs = angle_encode(X_all_processed, args.qubits)
                kernels = prepare_kernel_set(Xs, maps, p_noise, seed, args.batch_size)
                dataset_cache[dname] = (kernels, y, X_raw)

            for target in dataset_names:
                df, summary = run_target_seed(
                    target,
                    dataset_cache,
                    train_indices,
                    test_indices,
                    seed,
                    meta_learners,
                )
                df["dataset"] = target
                df["noise"] = noise_name
                df["noise_p"] = p_noise
                df["seed"] = seed
                # Summary values are repeated on each row to simplify grouping.
                df["best_individual_f1"] = summary["best_individual_f1"]
                df["uniform_f1"] = summary["uniform_f1"]
                df["adaptive_complementarity_f1"] = summary["adaptive_complementarity_f1"]
                df["adaptive_complementarity_gain"] = summary["adaptive_complementarity_gain"]
                df["mean_pairwise_complementarity"] = summary["mean_pairwise_complementarity"]
                df["mean_average_complementarity"] = summary["mean_average_complementarity"]
                all_results.append(df)

                # Independent classical RBF baseline, evaluated on the same raw
                # dataset split used above.
                X_raw, y = dataset_raw[target]
                tr = train_indices[target]
                te = test_indices[target]
                clf = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("svc", SVC(kernel="rbf", C=1.0))])
                clf.fit(X_raw[tr], y[tr])
                cm = metric_dict(y[te], clf.predict(X_raw[te]))
                classical = pd.DataFrame([{
                    "model": "Classical RBF SVM",
                    "selection": "none",
                    **cm,
                    **{f"weight_{n}": np.nan for n in FEATURE_MAP_NAMES},
                    "dataset": target,
                    "noise": noise_name,
                    "noise_p": p_noise,
                    "seed": seed,
                    "best_individual_f1": summary["best_individual_f1"],
                    "uniform_f1": summary["uniform_f1"],
                    "adaptive_complementarity_f1": summary["adaptive_complementarity_f1"],
                    "adaptive_complementarity_gain": summary["adaptive_complementarity_gain"],
                    "mean_pairwise_complementarity": summary["mean_pairwise_complementarity"],
                    "mean_average_complementarity": summary["mean_average_complementarity"],
                }])
                all_results.append(classical)

    results = pd.concat(all_results, ignore_index=True)

    # ------------------------------------------------------------------
    # Numerical results are printed ONLY here, at the bottom of execution.
    # Only image files are written to disk.
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("AMQKE-v25 NUMERICAL RESULTS")
    print("=" * 100)

    display_cols = [
        "dataset", "noise", "model", "selection",
        "accuracy", "precision", "recall", "f1",
    ]
    numeric_view = results[display_cols].copy()
    print("\n--- Per-run results ---")
    print(numeric_view.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Correct aggregation: selection is a per-run outcome, not a separate
    # statistical condition. Therefore it is intentionally excluded from the
    # primary Mean ± STD grouping for Contribution+Selection methods.
    aggregate = results.groupby(
        ["dataset", "noise", "model"],
        as_index=False,
    ).agg(
        n=("f1", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
    )

    print("\n--- Mean ± STD (selection-agnostic primary aggregation) ---")
    for _, r in aggregate.iterrows():
        print(
            f"{r['dataset']:>16} | {r['noise']:>9} | {r['model']:<32} | n={int(r['n']):d} | "
            f"Acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f} | "
            f"Prec={r['precision_mean']:.4f}±{r['precision_std']:.4f} | "
            f"Rec={r['recall_mean']:.4f}±{r['recall_std']:.4f} | "
            f"F1={r['f1_mean']:.4f}±{r['f1_std']:.4f}"
        )

    # Selection frequencies are reported separately, preserving every run.
    selection_rows = results[results["model"].str.startswith("Contribution+Selection")].copy()
    print("\n--- Dynamic selection frequencies ---")
    if selection_rows.empty:
        print("No dynamic-selection rows available.")
    else:
        sf = (
            selection_rows.groupby(["dataset", "noise", "model", "selection"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        sf["total"] = sf.groupby(["dataset", "noise", "model"])["count"].transform("sum")
        sf["frequency_pct"] = 100.0 * sf["count"] / sf["total"]
        for _, r in sf.sort_values(["dataset", "noise", "model", "frequency_pct"], ascending=[True, True, True, False]).iterrows():
            print(
                f"{r['dataset']:>16} | {r['noise']:>9} | {r['model']:<34} | "
                f"selection={r['selection']:<20} | count={int(r['count']):d} | freq={r['frequency_pct']:.1f}%"
            )

    # Noise-aware meta diagnostics: average predicted contribution and weight.
    print("\n--- Cross-dataset meta-learning diagnostics (mean over seeds) ---")
    adaptive_rows = results[
        (results["model"].str.startswith("Contribution-Aware"))
        & (results["selection"] == "none")
    ].copy()
    if adaptive_rows.empty:
        print("No contribution-aware rows available.")
    else:
        weight_cols = [f"weight_{n}" for n in FEATURE_MAP_NAMES]
        pred_cols = [f"pred_contribution_{n}" for n in FEATURE_MAP_NAMES]
        diag_group = adaptive_rows.groupby(["dataset", "noise", "model"], as_index=False)[weight_cols + pred_cols].mean()
        for _, r in diag_group.iterrows():
            learner = r["model"].replace("Contribution-Aware ", "")
            weights_text = ", ".join(f"{n}={r[f'weight_{n}']:.4f}" for n in FEATURE_MAP_NAMES)
            contrib_text = ", ".join(f"{n}={r[f'pred_contribution_{n}']:.4f}" for n in FEATURE_MAP_NAMES)
            print(
                f"Target={r['dataset']:<16} | Noise={r['noise']:<9} | Meta={learner:<13} | "
                f"PredictedContribution({contrib_text}) | Weights({weights_text})"
            )

    if not args.skip_plots:
        save_basic_figures(results, args.output_dir)
        print("\n--- Saved files ---")
        for image in sorted(args.output_dir.glob("*.png")):
            print(image.name)

    print("\n" + "=" * 100)
    print("END OF NUMERICAL OUTPUT")
    print("=" * 100)


if __name__ == "__main__":
    # Jupyter/IPython injects its own CLI arguments; ignore them.
    import sys
    running_in_ipykernel = "ipykernel" in sys.modules or "IPython" in sys.modules
    main([] if running_in_ipykernel else None)
