"""
Intelligent Wearable Insole System — Machine Learning Pipeline
==============================================================
Manuscript: "An intelligent wearable insole system for machine learning-based
             detection of high-risk load-lifting postures"
Journal   : Scientific Reports

Description
-----------
This script trains and evaluates five supervised classifiers (LR, SVM, KNN,
DT, RF) for binary classification of low-risk (Label 1) vs. high-risk
(Label 2) lifting postures using a 13-dimensional feature vector:
  - FSR1–FSR12  : bilateral plantar-pressure readings (calibrated, N)
  - TrunkAngle  : IMU-derived sagittal trunk flexion angle (degrees)

Ground-truth labels were derived from the UTAH back compressive force method
(threshold: 700 lbs / ≈ 3114 N).

Study Design
------------
- Phase 1 : 23 male participants × 37 conditions (36 lifting + 1 neutral)
            × 3-second data window at 5 Hz → 15 samples per condition
            → 851 total patterns after trimming (first/last second removed).
- Train/test split : 18 subjects (78 %) for training, 5 subjects (22 %) for
                     independent testing (participant-wise, no data leakage).
- Cross-validation : 5-fold stratified CV on the training subset.
- Feature selection: incremental addition based on Random Forest MDI ranking.

Authors
-------
M. Vafadar, A.H. Jafari, F. Karbasi, E. Ghaffari

Ethics Approval: IR.IUMS.REC.1402.966
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH           = "datafinal.xlsx"   # path to the data file (same folder or full path)
SEED                = 42
TRAIN_SUBJECT_COUNT = 18                 # 78 % of 23 subjects for training
CV_SPLITS           = 5                  # 5-fold stratified cross-validation (as reported)

# 13-dimensional feature vector (12 FSR channels + trunk flexion angle)
FEATURES = [
    "FSR1", "FSR2", "FSR3", "FSR4", "FSR5", "FSR6",
    "FSR7", "FSR8", "FSR9", "FSR10", "FSR11", "FSR12",
    "TrunkAngle",
]
# ──────────────────────────────────────────────────────────────────────────────


def load_dataset(path: str) -> pd.DataFrame:
    """Load all subject sheets from the Excel workbook.

    Each sheet represents one participant. Rows are shuffled within each sheet
    before concatenation to prevent any ordering artefacts.
    """
    xls = pd.ExcelFile(path)
    frames = []
    for i, sheet in enumerate(xls.sheet_names):
        df = pd.read_excel(path, sheet_name=sheet)
        rng = np.random.default_rng(SEED + i * 999)
        df = df.sample(frac=1, random_state=int(rng.integers(0, int(1e9)))).reset_index(drop=True)
        df["Subject"] = sheet
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Retain only the 13 feature columns, Label, and Subject; drop NaNs."""
    df = df[FEATURES + ["Label", "Subject"]].copy()
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce")
    df = df.dropna().reset_index(drop=True)
    df["Label"] = df["Label"].astype(int)
    return df


def normalize_per_subject(df: pd.DataFrame) -> pd.DataFrame:
    """Apply per-subject z-score normalization using StandardScaler.

    Each subject's sensor readings are independently scaled to mean = 0,
    std = 1. This removes inter-subject variability in absolute sensor
    magnitude (e.g. due to body weight differences) while preserving
    intra-subject postural patterns. Normalization is applied before the
    train/test split to reflect a realistic per-device calibration scenario
    in which a user-specific baseline is established before deployment.
    """
    df = df.copy()
    for subject in df["Subject"].unique():
        mask = df["Subject"] == subject
        scaler = StandardScaler()
        df.loc[mask, FEATURES] = scaler.fit_transform(df.loc[mask, FEATURES])
    return df


def split_subjects(df: pd.DataFrame):
    """Randomly assign 18 subjects to training and 5 to independent testing."""
    subjects = df["Subject"].unique().copy()
    np.random.default_rng(SEED).shuffle(subjects)
    return subjects[:TRAIN_SUBJECT_COUNT], subjects[TRAIN_SUBJECT_COUNT:]


def build_split(df: pd.DataFrame, train_subs, test_subs):
    """Return (X_train, X_test, y_train, y_test) DataFrames/Series."""
    tr = df[df["Subject"].isin(train_subs)].sample(frac=1, random_state=SEED)
    te = df[df["Subject"].isin(test_subs)].sample(frac=1, random_state=SEED + 1)
    return (
        tr[FEATURES].reset_index(drop=True),
        te[FEATURES].reset_index(drop=True),
        tr["Label"].reset_index(drop=True),
        te["Label"].reset_index(drop=True),
    )


def feature_ranking(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Rank features by Random Forest Mean Decrease in Impurity (MDI)."""
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED)
    rf.fit(X_train, y_train)
    return (
        pd.DataFrame({"feature": X_train.columns, "importance": rf.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# ── Model Definitions ─────────────────────────────────────────────────────────
# All hyperparameter grids are searched via GridSearchCV with 5-fold CV.
# LR  : L2 regularisation, newton-cg solver, best C from {0.01, 0.1, 1, 10}
#        → best C = 0.1 (as reported in manuscript)
# SVM : RBF kernel, best C/γ from grid → best C=1, γ=0.1 (as reported)
# KNN : Euclidean distance-weighted, best k from {1..5} → best k=3 (reported)
# DT  : Unrestricted depth, min_samples_split=5, min_samples_leaf=2 (reported)
# RF  : 100 trees, min_samples_split=5, min_samples_leaf=2 (reported)
MODEL_CONFIGS = {
    "LR": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="l2",
                solver="newton-cg",
                max_iter=5000,
                random_state=SEED,
            )),
        ]),
        "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
    },
    "SVM": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=SEED)),
        ]),
        "param_grid": {
            "clf__C":     [0.1, 1, 10],
            "clf__gamma": [0.01, 0.1, 1],
        },
    },
    "KNN": {
        "estimator": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(metric="euclidean", weights="distance")),
        ]),
        "param_grid": {"clf__n_neighbors": [1, 2, 3, 4, 5]},
    },
    "DT": {
        "estimator": DecisionTreeClassifier(random_state=SEED),
        "param_grid": {
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
        },
    },
    "RF": {
        "estimator": RandomForestClassifier(n_estimators=100, random_state=SEED),
        "param_grid": {
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
        },
    },
}
# ──────────────────────────────────────────────────────────────────────────────


def get_proba(model, X: pd.DataFrame):
    """Return probability scores for Label 2 (high-risk) for AUC computation."""
    if hasattr(model, "predict_proba"):
        proba     = model.predict_proba(X)
        class_list = list(model.classes_)
        if 2 not in class_list:
            return None
        return proba[:, class_list.index(2)]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute classification metrics from a binary confusion matrix.

    Labels: 1 = low-risk (negative class), 2 = high-risk (positive class).
    TP = high-risk correctly identified; TN = low-risk correctly identified.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2])
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall for high-risk
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # recall for low-risk

    auc = np.nan
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            pass

    hr_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    hr_f1   = (2 * hr_prec * sensitivity / (hr_prec + sensitivity)
               if (hr_prec + sensitivity) > 0 else 0.0)

    lr_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    lr_f1   = (2 * lr_prec * specificity / (lr_prec + specificity)
               if (lr_prec + specificity) > 0 else 0.0)

    return dict(
        accuracy=accuracy, sensitivity=sensitivity, specificity=specificity, auc=auc,
        hr_precision=hr_prec, hr_recall=sensitivity, hr_f1=hr_f1,
        lr_precision=lr_prec, lr_recall=specificity, lr_f1=lr_f1,
    )


def run_incremental(
    name: str,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_test:  pd.DataFrame, y_test:  pd.Series,
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Train each model incrementally from k=1 to k=13 features.

    At each step, features are added in MDI importance order. Hyperparameters
    are optimised via GridSearchCV with 5-fold stratified CV on the training
    set. Final metrics are reported on the independent test set.
    """
    feature_order = ranking["feature"].tolist()
    cfg = MODEL_CONFIGS[name]
    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=SEED)
    records = []

    for k in range(1, len(feature_order) + 1):
        cols = feature_order[:k]

        gs = GridSearchCV(
            estimator=cfg["estimator"],
            param_grid=cfg["param_grid"],
            cv=skf,
            scoring="accuracy",
            refit=True,
            n_jobs=-1,
        )
        gs.fit(X_train[cols], y_train)
        best_model = gs.best_estimator_

        y_pred = best_model.predict(X_test[cols])
        y_prob = get_proba(best_model, X_test[cols])

        m = compute_metrics(y_test, y_pred, y_prob)
        m.update({
            "k": k,
            "features":    ", ".join(cols),
            "best_params": str(gs.best_params_),
            "cv_accuracy": gs.best_score_,
        })
        records.append(m)

        print(
            f"  {name} | k={k:2d} | "
            f"acc={m['accuracy']:.3f}  sens={m['sensitivity']:.3f}  "
            f"spec={m['specificity']:.3f}  auc={m['auc']:.3f} | "
            f"HR(P={m['hr_precision']:.3f} R={m['hr_recall']:.3f} F1={m['hr_f1']:.3f})  "
            f"LR(P={m['lr_precision']:.3f} R={m['lr_recall']:.3f} F1={m['lr_f1']:.3f})"
        )

    return pd.DataFrame(records)


def build_best_per_k_results(all_results: dict, models: list) -> pd.DataFrame:
    """Select the best-performing model (by accuracy then AUC) at each k."""
    rows = []
    for k in range(1, len(FEATURES) + 1):
        candidates = [
            (name, all_results[name][all_results[name]["k"] == k].iloc[0])
            for name in models
        ]
        best_name, best_row = max(candidates, key=lambda x: (x[1]["accuracy"], x[1]["auc"]))
        entry = best_row.to_dict()
        entry["best_model"] = best_name
        rows.append(entry)

    df    = pd.DataFrame(rows)
    front = [
        "k", "best_model", "features", "accuracy", "sensitivity", "specificity",
        "auc", "hr_precision", "hr_recall", "hr_f1",
        "lr_precision", "lr_recall", "lr_f1", "cv_accuracy", "best_params",
    ]
    return df[[c for c in front if c in df.columns]]


def print_summary(all_results: dict, k: int = 13) -> None:
    """Print a formatted performance table for all models at k features."""
    header = (
        f"{'Model':<6} | "
        f"{'HR_Prec':>8} {'HR_Rec':>7} {'HR_F1':>6} | "
        f"{'LR_Prec':>8} {'LR_Rec':>7} {'LR_F1':>6} | "
        f"{'Acc':>6} {'Sens':>6} {'Spec':>6} {'AUC':>6}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}")
    print(f"  k={k} features — Independent Test-Set Performance")
    print(sep)
    print(header)
    print("-" * len(header))

    for name, dfm in all_results.items():
        row = dfm[dfm["k"] == k].iloc[0]
        pct = lambda x: f"{x * 100:.1f}%"
        print(
            f"{name:<6} | "
            f"{pct(row['hr_precision']):>8} {pct(row['hr_recall']):>7} {pct(row['hr_f1']):>6} | "
            f"{pct(row['lr_precision']):>8} {pct(row['lr_recall']):>7} {pct(row['lr_f1']):>6} | "
            f"{pct(row['accuracy']):>6} {pct(row['sensitivity']):>6} "
            f"{pct(row['specificity']):>6} {row['auc']:>6.3f}"
        )
    print(sep)


def main() -> None:
    print("=" * 70)
    print("  Intelligent Wearable Insole — ML Classification Pipeline")
    print("=" * 70)

    # ── Load & clean ───────────────────────────────────────────────────────
    print("\nLoading dataset …")
    df = load_dataset(DATA_PATH)
    df = clean_dataset(df)
    print(f"  Shape    : {df.shape}")
    print(f"  Subjects : {df['Subject'].nunique()}")
    print(f"  Labels   :\n{df['Label'].value_counts().to_string()}\n")

    # ── Per-subject normalization ──────────────────────────────────────────
    print("Applying per-subject z-score normalization …")
    df = normalize_per_subject(df)
    print("  ✓ Each subject's features scaled to mean=0, std=1\n")

    # ── Train / test split ─────────────────────────────────────────────────
    train_subs, test_subs = split_subjects(df)
    X_train, X_test, y_train, y_test = build_split(df, train_subs, test_subs)
    print(f"Train : {len(X_train)} samples ({len(train_subs)} subjects)")
    print(f"Test  : {len(X_test)} samples ({len(test_subs)} subjects)\n")

    # ── Feature importance ─────────────────────────────────────────────────
    print("Computing Random Forest MDI feature importance …")
    ranking = feature_ranking(X_train, y_train)
    print(ranking.to_string(index=False), "\n")

    # ── Incremental model evaluation ───────────────────────────────────────
    models      = ["RF", "KNN", "DT", "SVM", "LR"]
    all_results = {}

    for name in models:
        print(f"\n{'=' * 70}")
        print(f"  {name}  —  GridSearchCV + {CV_SPLITS}-fold stratified CV, k = 1 → {len(FEATURES)}")
        print(f"{'=' * 70}")
        all_results[name] = run_incremental(
            name, X_train, y_train, X_test, y_test, ranking
        )

    # ── Summary tables ─────────────────────────────────────────────────────
    print_summary(all_results, k=len(FEATURES))

    mean_cols = [
        "accuracy", "sensitivity", "specificity", "auc",
        "hr_precision", "hr_recall", "hr_f1",
        "lr_precision", "lr_recall", "lr_f1",
    ]
    summary_by_k = pd.DataFrame({"k": range(1, len(FEATURES) + 1)})
    for col in mean_cols:
        summary_by_k[f"mean_{col}"] = [
            np.mean([
                all_results[m][all_results[m]["k"] == k][col].values[0]
                for m in models
            ])
            for k in range(1, len(FEATURES) + 1)
        ]

    print("\n\nMean performance across all 5 models (accuracy / sensitivity / specificity / AUC):")
    print(
        summary_by_k[["k", "mean_accuracy", "mean_sensitivity", "mean_specificity", "mean_auc"]]
        .to_string(index=False)
    )

    best_per_k = build_best_per_k_results(all_results, models)
    print("\n\nBest model per k (accuracy / sensitivity / specificity / AUC):")
    print(
        best_per_k[["k", "best_model", "accuracy", "sensitivity", "specificity", "auc"]]
        .to_string(index=False)
    )

    # ── Save outputs ───────────────────────────────────────────────────────
    out = "results"
    os.makedirs(out, exist_ok=True)

    output_files = {
        "feature_importance.csv": ranking,
        "mean_performance_by_k.csv": summary_by_k,
        "best_model_per_k.csv": best_per_k,
        **{f"results_{name}.csv": dfm for name, dfm in all_results.items()},
    }

    print(f"\nSaving outputs to: {out}/")
    for filename, dataframe in output_files.items():
        filepath = os.path.join(out, filename)
        dataframe.to_csv(filepath, index=False)
        print(f"  ✓ {filename}  ({len(dataframe)} rows × {len(dataframe.columns)} cols)")

    print("\nDone.")


if __name__ == "__main__":
    main()
