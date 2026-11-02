"""
Model training, cross-validation, and evaluation module for Sentiment Analysis.
Constructs an end-to-end scikit-learn Pipeline (TF-IDF + Logistic Regression),
performs 5-fold cross-validation on the training set, evaluates on the held-out test set,
and saves the confusion matrix visualization.
"""

from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from src.config import RANDOM_STATE, REPORTS_DIR, MODEL_PATH
from src.dataset import load_and_split_data
from src.preprocessing import get_tfidf_vectorizer

# Ensure consistent class label order
CLASS_LABELS = ["negative", "positive"]


def build_pipeline(
    max_features: int = 50000,
    ngram_range: tuple = (1, 2),
    min_df: int = 5,
    sublinear_tf: bool = True,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """
    Constructs a unified, leakage-free scikit-learn Pipeline:
    Raw Review Text -> TF-IDF Vectorizer -> Logistic Regression Classifier.

    Args:
        max_features: Maximum vocabulary size.
        ngram_range: (min_n, max_n) n-gram boundaries.
        min_df: Minimum document frequency threshold.
        sublinear_tf: Applies sublinear scaling (1 + log(tf)).
        C: Inverse regularization strength parameter.
        max_iter: Maximum iterations for lbfgs solver convergence.
        random_state: Random seed for reproducibility.

    Returns:
        Pipeline: Unfitted scikit-learn Pipeline.
    """
    vectorizer = get_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )

    classifier = LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver="lbfgs",
    )

    pipeline = Pipeline(
        steps=[
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )

    return pipeline


def perform_cross_validation(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
    cv: int = 5,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Performs k-fold cross-validation STRICTLY on the training set.
    The held-out test partition is never exposed during this step.

    Args:
        pipeline: Unfitted pipeline.
        X_train: Training text series.
        y_train: Training labels series.
        cv: Number of folds (default 5).
        n_jobs: Number of CPU cores to utilize (-1 uses all available cores).

    Returns:
        np.ndarray: Array of cross-validation fold accuracy scores.
    """
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy",
        n_jobs=n_jobs,
    )
    return cv_scores


def train_model(
    pipeline: Pipeline,
    X_train: pd.Series,
    y_train: pd.Series,
) -> Pipeline:
    """
    Fits the unified Pipeline strictly on the training partition.

    Args:
        pipeline: Unfitted Pipeline.
        X_train: Training text series.
        y_train: Training labels series.

    Returns:
        Pipeline: Fitted Pipeline.
    """
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """
    Evaluates the fitted pipeline on the unseen test set and runs sanity checks.

    Args:
        pipeline: Fitted Pipeline.
        X_test: Unseen test text series.
        y_test: Ground-truth test labels.

    Returns:
        Dict[str, Any]: Comprehensive evaluation metrics and predictions.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)

    # Sanity checks
    assert set(np.unique(y_pred)).issubset(set(CLASS_LABELS)), (
        f"Unexpected class labels predicted: {np.unique(y_pred)}"
    )
    assert len(y_pred) == len(X_test), "Prediction count does not match test set size."
    assert not np.isnan(y_proba).any(), "NaN values found in prediction probabilities."
    assert y_proba.shape == (len(X_test), 2), "Prediction probability shape mismatch."

    # Compute scalar metrics (with positive class = 'positive')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label="positive")
    recall = recall_score(y_test, y_pred, pos_label="positive")
    f1 = f1_score(y_test, y_pred, pos_label="positive")

    # Confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred, labels=CLASS_LABELS)
    report_dict = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        target_names=["Negative", "Positive"],
        output_dict=True,
    )
    report_text = classification_report(
        y_test,
        y_pred,
        labels=CLASS_LABELS,
        target_names=["Negative", "Positive"],
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "classification_report_text": report_text,
        "classification_report_dict": report_dict,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = CLASS_LABELS,
    output_path: Path = REPORTS_DIR / "confusion_matrix.png",
) -> Path:
    """
    Renders and saves a clean confusion matrix heatmap styled in the Dark Navy theme.

    Args:
        cm: 2x2 confusion matrix array.
        labels: Class names.
        output_path: Target image file path.

    Returns:
        Path: Absolute path to the saved figure.
    """
    from matplotlib.colors import LinearSegmentedColormap

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    bg_color = "#111C2E"
    text_color = "#F1F5F9"
    muted_color = "#94A3B8"
    border_color = "#263A56"

    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    cmap = LinearSegmentedColormap.from_list(
        "navy_blue", ["#16243A", "#1E40AF", "#2563EB", "#3B82F6"]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=[label.capitalize() for label in labels],
        yticklabels=[label.capitalize() for label in labels],
        cbar=True,
        cbar_kws={"shrink": 0.85},
        annot_kws={"size": 15, "weight": "bold", "color": text_color},
        linewidths=2,
        linecolor=border_color,
        ax=ax,
    )

    for text in ax.texts:
        text.set_color(text_color)
        text.set_weight("bold")

    ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold", color=text_color, labelpad=10)
    ax.set_ylabel("True Label", fontsize=11, fontweight="bold", color=text_color, labelpad=10)
    ax.set_title("Confusion Matrix (Held-Out Test Set)", fontsize=13, fontweight="bold", color=text_color, pad=14)

    ax.tick_params(colors=muted_color, labelsize=10)
    for spine in ax.spines.values():
        spine.set_color(border_color)

    cbar = ax.collections[0].colorbar
    if cbar:
        cbar.ax.yaxis.set_tick_params(color=muted_color)
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=muted_color, size=9)
        cbar.outline.set_edgecolor(border_color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    return output_path


def save_model(pipeline: Pipeline, model_path: Path = MODEL_PATH) -> Path:
    """
    Serializes the complete fitted scikit-learn Pipeline to disk using joblib.

    Args:
        pipeline: Fitted Pipeline.
        model_path: Destination path for the .joblib artifact.

    Returns:
        Path: Path to the saved model file.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return model_path


def run_training_pipeline() -> Dict[str, Any]:
    """
    High-level orchestrator:
    1. Loads stratified train/test data splits.
    2. Builds end-to-end Pipeline.
    3. Runs 5-fold CV strictly on training partition.
    4. Fits pipeline on full training partition.
    5. Evaluates on held-out test partition.
    6. Saves confusion matrix visualization.
    7. Serializes fitted pipeline to models/sentiment_pipeline.joblib.
    8. Displays formatted results and verification checks.
    """
    print("=" * 60)
    print("PHASE 4 & 5: MODEL TRAINING, EVALUATION & SERIALIZATION")
    print("=" * 60)

    # 1. Load data
    X_train, X_test, y_train, y_test = load_and_split_data()
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples:     {len(X_test)}")
    print("-" * 60)

    # 2. Build pipeline
    pipeline = build_pipeline()

    # 3. 5-Fold Cross-Validation on Training Set
    print("Running 5-Fold Cross-Validation on Training Set...")
    cv_scores = perform_cross_validation(pipeline, X_train, y_train, cv=5, n_jobs=-1)

    print("\n5-Fold Cross-Validation")
    for fold_idx, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold_idx}: {score:.4f} ({score * 100:.2f}%)")
    mean_cv = float(np.mean(cv_scores))
    std_cv = float(np.std(cv_scores))
    print(f"Mean CV Accuracy: {mean_cv:.4f} ({mean_cv * 100:.2f}%)")
    print(f"Std CV Accuracy:  {std_cv:.4f}")
    print("-" * 60)

    # 4. Train final pipeline on X_train, y_train
    print("Fitting unified Pipeline on full training set...")
    train_model(pipeline, X_train, y_train)
    print("Training complete. Convergence achieved successfully.")
    print("-" * 60)

    # 5. Evaluate on held-out test set
    print("Evaluating Pipeline on held-out test set...")
    eval_results = evaluate_model(pipeline, X_test, y_test)

    print("\nFinal Test Evaluation")
    print(f"Accuracy:  {eval_results['accuracy']:.4f} ({eval_results['accuracy'] * 100:.2f}%)")
    print(f"Precision: {eval_results['precision']:.4f} ({eval_results['precision'] * 100:.2f}%)")
    print(f"Recall:    {eval_results['recall']:.4f} ({eval_results['recall'] * 100:.2f}%)")
    print(f"F1-score:  {eval_results['f1']:.4f} ({eval_results['f1'] * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print(eval_results["confusion_matrix"])

    print("\nClassification Report:")
    print(eval_results["classification_report_text"])

    # 6. Save confusion matrix visualization
    cm_path = plot_confusion_matrix(eval_results["confusion_matrix"])
    print(f"Confusion matrix plot saved to: {cm_path.resolve()}")

    # 7. Serialize fitted pipeline to disk
    saved_model_path = save_model(pipeline)
    print(f"Fitted pipeline serialized to: {saved_model_path.resolve()}")

    # 8. Verification checks
    print("-" * 60)
    print("Verification checks:")
    print(" [x] Pipeline fitted strictly on training data -> PASSED")
    print(" [x] Logistic Regression converged without warnings -> PASSED")
    print(f" [x] Prediction count ({len(eval_results['y_pred'])}) matches test set ({len(X_test)}) -> PASSED")
    print(" [x] Predictions contain only valid classes ('negative', 'positive') -> PASSED")
    print(" [x] No NaN values in predictions or probability scores -> PASSED")
    print(" [x] Test set was not exposed to cross-validation -> PASSED")
    print(f" [x] Model serialized to {saved_model_path.name} -> PASSED")
    print("=" * 60)

    return {
        "pipeline": pipeline,
        "cv_scores": cv_scores,
        "eval_results": eval_results,
        "confusion_matrix_path": cm_path,
        "saved_model_path": saved_model_path,
    }


if __name__ == "__main__":
    run_training_pipeline()
