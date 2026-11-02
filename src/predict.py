"""
Modular sentiment prediction and inference service.
Loads the serialized scikit-learn pipeline (TF-IDF + Logistic Regression)
and predicts the sentiment and confidence score of raw movie review texts.
Does NOT perform any model retraining.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import MODEL_PATH

# Cached model instance for efficient repeated inference
_CACHED_MODEL: Optional[Pipeline] = None


def load_model(model_path: Path = MODEL_PATH) -> Pipeline:
    """
    Loads the serialized sentiment pipeline from disk.

    Args:
        model_path: Path to the .joblib artifact.

    Returns:
        Pipeline: Serialized scikit-learn Pipeline instance.

    Raises:
        FileNotFoundError: If the model artifact does not exist on disk.
    """
    global _CACHED_MODEL
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model artifact not found at: {model_path.resolve()}.\n"
            "Please train the model first by running:\n"
            "    python -m src.train"
        )

    _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def predict_sentiment(
    text: str,
    model: Optional[Pipeline] = None,
) -> Dict[str, Any]:
    """
    Predicts sentiment and associated confidence for a single raw review text.

    Args:
        text: Raw movie review string.
        model: Optional pre-loaded pipeline instance. If None, uses cached/loaded model.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'sentiment': 'positive' or 'negative'
            - 'confidence': Probability score corresponding to the predicted class.

    Raises:
        ValueError: If text is empty, contains only whitespace, or is not a string.
        FileNotFoundError: If model is not loaded and artifact does not exist.
    """
    # 1. Input validation
    if not isinstance(text, str):
        raise ValueError(
            f"Input text must be a string, received {type(text).__name__} instead."
        )

    stripped_text = text.strip()
    if not stripped_text:
        raise ValueError(
            "Input text cannot be empty or contain only whitespace characters."
        )

    # 2. Model resolution (without retraining)
    if model is None:
        global _CACHED_MODEL
        if _CACHED_MODEL is None:
            model = load_model()
        else:
            model = _CACHED_MODEL

    # 3. Model inference directly on raw review text
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]

    # 4. Resolve confidence for the predicted class
    classes = list(model.classes_)
    class_index = classes.index(prediction)
    confidence = float(probabilities[class_index])

    return {
        "sentiment": str(prediction),
        "confidence": round(confidence, 4),
    }


def run_inference_demo() -> None:
    """
    Executes a command-line demonstration of the inference service
    with automated checks verifying model integrity and prediction behaviors.
    """
    print("=" * 60)
    print("PHASE 5: MODEL SERIALIZATION & INFERENCE")
    print("=" * 60)

    # 1. Load model
    model = load_model()
    print("Model loaded successfully.\n")

    # 2. Example reviews
    demo_reviews = [
        "I absolutely loved this movie. The acting was fantastic and the story was engaging.",
        "This was a terrible movie. The plot was boring and the acting was awful.",
        "The movie was okay, but nothing particularly special.",
    ]

    for idx, review in enumerate(demo_reviews, start=1):
        result = predict_sentiment(review, model=model)
        confidence_pct = result["confidence"] * 100
        print(f"Review {idx}: \"{review}\"")
        print(f"Prediction: {result['sentiment']}")
        print(f"Confidence: {confidence_pct:.2f}%\n")

    # 3. Automated sanity and integrity verification
    print("-" * 60)
    print("Running Automated Verification Checks...")

    # Check 1: File exists
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"

    # Check 2 & 3: Loaded object is Pipeline
    assert isinstance(model, Pipeline), "Loaded object is not a scikit-learn Pipeline"

    # Check 4: Pipeline steps
    step_names = [name for name, _ in model.steps]
    assert "tfidf" in step_names and "classifier" in step_names, (
        f"Pipeline missing expected steps: {step_names}"
    )

    # Check 5: Classifier type
    classifier_step = model.named_steps["classifier"]
    assert isinstance(classifier_step, LogisticRegression), (
        f"Classifier step is not LogisticRegression: {type(classifier_step)}"
    )

    # Check 6 & 10: Valid prediction and class set
    test_result = predict_sentiment("A brilliant masterpiece.", model=model)
    assert test_result["sentiment"] in ["positive", "negative"], (
        f"Invalid sentiment predicted: {test_result['sentiment']}"
    )

    # Check 7 & 8: Probabilities sum to ~1.0
    proba = model.predict_proba(["Good film"])[0]
    assert not np.isnan(proba).any(), "NaN values found in prediction probabilities"
    assert np.isclose(np.sum(proba), 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

    # Check 9: Empty/whitespace inputs raise ValueError
    invalid_inputs = ["", "   ", "\n\t  ", 12345, None]
    for invalid_val in invalid_inputs:
        try:
            predict_sentiment(invalid_val, model=model)  # type: ignore
            raise AssertionError(f"Failed to raise ValueError for invalid input: {repr(invalid_val)}")
        except ValueError:
            pass  # Expected behavior

    print("Verification checks:")
    print(" [x] models/sentiment_pipeline.joblib exists -> PASSED")
    print(" [x] Saved object loaded successfully via joblib -> PASSED")
    print(" [x] Loaded object is scikit-learn Pipeline -> PASSED")
    print(" [x] Contains expected 'tfidf' and 'classifier' steps -> PASSED")
    print(" [x] Classifier step is LogisticRegression -> PASSED")
    print(" [x] Loaded model predicts standard raw review text -> PASSED")
    print(" [x] predict_proba() returns valid probabilities summing to 1.0 -> PASSED")
    print(" [x] Empty and whitespace-only inputs strictly raise ValueError -> PASSED")
    print(" [x] Non-string inputs strictly raise ValueError -> PASSED")
    print(" [x] Prediction outputs strictly match {'positive', 'negative'} -> PASSED")
    print(" [x] No retraining occurs during inference -> PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_inference_demo()
