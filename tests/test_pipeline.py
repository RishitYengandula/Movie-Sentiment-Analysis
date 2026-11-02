"""
Sanity and unit tests for the sentiment pipeline, preprocessing, and inference engine.
"""

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import MODEL_PATH
from src.preprocessing import clean_text
from src.predict import load_model, predict_sentiment


def test_clean_text_edge_cases():
    """Verifies text normalization strips HTML, symbols, and redundant spaces."""
    assert clean_text("<br /><p>Great film!</p>") == "great film"
    assert clean_text("AWESOME!!! Loved it...") == "awesome loved it"
    assert clean_text("   Too    much   space   ") == "too much space"
    assert clean_text(12345) == ""  # type: ignore
    assert clean_text(None) == ""   # type: ignore


def test_model_loading():
    """Verifies the serialized model exists, loads cleanly, and retains pipeline structure."""
    assert MODEL_PATH.exists(), f"Model artifact missing at {MODEL_PATH}"
    model = load_model()
    assert isinstance(model, Pipeline), "Loaded artifact is not a Pipeline"
    assert "tfidf" in model.named_steps
    assert "classifier" in model.named_steps
    assert isinstance(model.named_steps["tfidf"], TfidfVectorizer)
    assert isinstance(model.named_steps["classifier"], LogisticRegression)


def test_predict_sentiment_valid():
    """Verifies sentiment prediction format and valid values on typical reviews."""
    positive_review = "An extraordinary cinematic masterpiece with phenomenal acting!"
    result = predict_sentiment(positive_review)
    assert isinstance(result, dict)
    assert "sentiment" in result
    assert "confidence" in result
    assert result["sentiment"] in ["positive", "negative"]
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_probabilities_and_confidence():
    """Verifies predict_proba returns probabilities that sum to 1.0 and match confidence."""
    model = load_model()
    review = "A fairly enjoyable watch with good characters."
    proba = model.predict_proba([review])[0]

    assert len(proba) == 2
    assert not np.isnan(proba).any()
    assert np.isclose(np.sum(proba), 1.0, atol=1e-5)

    res = predict_sentiment(review, model=model)
    classes = list(model.classes_)
    expected_confidence = float(proba[classes.index(res["sentiment"])])
    assert np.isclose(res["confidence"], expected_confidence, atol=1e-3)


@pytest.mark.parametrize("invalid_input", ["", "   ", "\t\n  ", 999, None, []])
def test_predict_invalid_inputs(invalid_input):
    """Verifies that empty, whitespace-only, or non-string inputs raise ValueError."""
    with pytest.raises(ValueError):
        predict_sentiment(invalid_input)  # type: ignore
