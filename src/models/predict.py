"""
Model training, evaluation, and inference utilities.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.features.engineering import FEATURE_COLUMNS, LABEL_MAP, INV_LABEL_MAP


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Class weights compensate for outcome imbalance (home wins are most common).
CLASS_WEIGHTS = {0: 1.0, 1: 1.45, 2: 1.3}

PARAM_GRID = {
    "n_estimators":     [200, 400, 600, 700],
    "max_depth":        [1, 2, 3, 4],
    "learning_rate":    [0.3, 0.05, 0.1],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

XGB_BASE_PARAMS = {
    "objective":   "multi:softprob",
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "num_class":   3,
    "random_state": 42,
}


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------

def prepare_training_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the engineered dataset into train / test sets.

    Args:
        df: DataFrame with FEATURE_COLUMNS and a FTR column.
        test_size: Fraction of data held out for evaluation.
        random_state: Seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df[FEATURE_COLUMNS].astype("float32")
    y = df["FTR"].map(LABEL_MAP)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _sample_weights(y: pd.Series) -> np.ndarray:
    return y.map(CLASS_WEIGHTS).values


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 5,
    cv: int = 3,
    random_state: int = 42,
) -> Any:
    """Train an XGBoost classifier with randomised hyperparameter search.

    Class weights are applied to address the home-win / draw / away-win
    imbalance in the training data.

    Args:
        X_train: Feature matrix (float32).
        y_train: Target labels (0=H, 1=D, 2=A).
        n_iter: Number of parameter combinations to try.
        cv: Number of cross-validation folds.
        random_state: Seed for reproducibility.

    Returns:
        The best fitted XGBClassifier found by the search.
    """
    from sklearn.model_selection import RandomizedSearchCV
    from xgboost import XGBClassifier

    base = XGBClassifier(**XGB_BASE_PARAMS)
    search = RandomizedSearchCV(
        base,
        param_distributions=PARAM_GRID,
        n_iter=n_iter,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        random_state=random_state,
        verbose=1,
    )
    sample_weights = _sample_weights(y_train)
    search.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"Best params : {search.best_params_}")
    print(f"CV accuracy : {search.best_score_:.4f}")
    return search.best_estimator_


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict:
    """Evaluate the model on the held-out test set.

    Args:
        model: Fitted XGBClassifier.
        X_test: Feature matrix for test matches.
        y_test: True labels for test matches.

    Returns:
        Dictionary with accuracy, classification_report, and confusion_matrix.
    """
    y_pred = model.predict(X_test)
    target_names = [INV_LABEL_MAP[i] for i in sorted(INV_LABEL_MAP)]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=target_names),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "target_names": target_names,
    }


def plot_confusion_matrix(eval_results: dict) -> "ConfusionMatrixDisplay":
    """Return a ConfusionMatrixDisplay from evaluate_model output."""
    disp = ConfusionMatrixDisplay(
        confusion_matrix=eval_results["confusion_matrix"],
        display_labels=eval_results["target_names"],
    )
    return disp


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model, path: Path) -> None:
    """Save the model to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))
    print(f"Model saved → {path}")


def load_model(path: Path):
    """Load a previously saved Random Forest model."""
    import joblib
    return joblib.load(str(Path(path)))


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Run inference and return labels + probabilities.

    Args:
        model: Fitted XGBClassifier.
        X: Feature matrix; NaN values are filled with 0 before scoring.

    Returns:
        DataFrame with columns: PredictedResult, Prob_H, Prob_D, Prob_A.
    """
    X_clean = X[FEATURE_COLUMNS].astype("float32").fillna(0.0)

    labels_numeric = model.predict(X_clean)
    probabilities = model.predict_proba(X_clean)

    prob_cols = [f"Prob_{INV_LABEL_MAP[i]}" for i in sorted(INV_LABEL_MAP)]
    result = pd.DataFrame(probabilities, columns=prob_cols, index=X.index)
    result.insert(0, "PredictedResult", [INV_LABEL_MAP[int(l)] for l in labels_numeric])
    return result
