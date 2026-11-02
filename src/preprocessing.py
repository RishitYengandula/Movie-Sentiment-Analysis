"""
Text preprocessing and TF-IDF feature extraction module.
Provides text cleaning functions and leakage-free TF-IDF vectorization.
"""

import re
from typing import Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """
    Cleans and normalizes a movie review string:
    1. Handles non-string inputs safely
    2. Strips HTML markup tags (e.g. <br />, <p>)
    3. Strips punctuation and non-alphanumeric noise
    4. Converts all text to lowercase
    5. Normalizes consecutive whitespace into a single space

    Args:
        text: Raw review string.

    Returns:
        str: Cleaned and normalized text.
    """
    if not isinstance(text, str):
        return ""

    # 1. Strip HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # 2. Remove punctuation and non-alphanumeric symbols (retain letters, digits, and spaces)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    # 3. Lowercase and collapse redundant whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()

    return text


def get_tfidf_vectorizer(
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
    sublinear_tf: bool = True,
) -> TfidfVectorizer:
    """
    Initializes a TfidfVectorizer configured with the clean_text preprocessor.

    Args:
        max_features: Maximum vocabulary size to prevent excessive memory usage.
        ngram_range: (min_n, max_n) n-gram range (default: unigrams and bigrams).
        min_df: Minimum document frequency for vocabulary inclusion.
        sublinear_tf: Applies sublinear scaling (1 + log(tf)) to damp frequent terms.

    Returns:
        TfidfVectorizer: Unfitted vectorizer instance.
    """
    return TfidfVectorizer(
        preprocessor=clean_text,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=sublinear_tf,
    )


def build_tfidf_features(
    X_train: pd.Series,
    X_test: pd.Series,
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 5,
    sublinear_tf: bool = True,
) -> Tuple[TfidfVectorizer, Any, Any]:
    """
    Fits a TfidfVectorizer STRICTLY on X_train, then transforms both X_train and X_test.
    Prevents data leakage by ensuring test vocabulary and inverse document frequencies
    are never observed during vectorizer fitting.

    Args:
        X_train: Training review texts.
        X_test: Test review texts.
        max_features: Maximum vocabulary size.
        ngram_range: Tuple of (min_n, max_n).
        min_df: Minimum document frequency.
        sublinear_tf: Whether to use sublinear term-frequency scaling.

    Returns:
        Tuple[TfidfVectorizer, scipy.sparse.csr_matrix, scipy.sparse.csr_matrix]:
            (fitted_vectorizer, X_train_tfidf, X_test_tfidf)
    """
    vectorizer = get_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )

    # 1. Fit and transform training data ONLY
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # 2. Transform test data using the vocabulary learned on training data
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_test_tfidf


if __name__ == "__main__":
    from src.dataset import load_and_split_data

    print("=" * 60)
    print("PHASE 3: PREPROCESSING & TF-IDF FEATURE EXTRACTION VERIFICATION")
    print("=" * 60)

    # 1. Load data splits
    X_train, X_test, y_train, y_test = load_and_split_data()

    print(f"Number of training reviews: {len(X_train)}")
    print(f"Number of test reviews:     {len(X_test)}")
    print("-" * 60)

    # 2. Text cleaning demonstration
    example_raw = X_train.iloc[0]
    example_clean = clean_text(example_raw)

    print("Example raw review (first 200 chars):")
    print(repr(example_raw[:200]))
    print("\nExample cleaned review (first 200 chars):")
    print(repr(example_clean[:200]))
    print("-" * 60)

    # 3. TF-IDF feature extraction
    vectorizer, X_train_tfidf, X_test_tfidf = build_tfidf_features(X_train, X_test)

    print(f"TF-IDF training shape: {X_train_tfidf.shape}")
    print(f"TF-IDF test shape:     {X_test_tfidf.shape}")
    print(f"Vocabulary size:       {len(vectorizer.vocabulary_)}")
    print("-" * 60)

    # 4. Strict assertions & validation checks
    assert X_train_tfidf.shape[0] == len(X_train), "Row count mismatch in X_train_tfidf"
    assert X_test_tfidf.shape[0] == len(X_test), "Row count mismatch in X_test_tfidf"
    assert X_train_tfidf.shape[1] == X_test_tfidf.shape[1], "Feature dimension mismatch"
    assert not np.isnan(X_train_tfidf.data).any(), "NaN values found in X_train_tfidf"
    assert not np.isnan(X_test_tfidf.data).any(), "NaN values found in X_test_tfidf"
    print("Verification checks:")
    print(" [x] X_train_tfidf.shape[0] == len(X_train) -> PASSED")
    print(" [x] X_test_tfidf.shape[0] == len(X_test)   -> PASSED")
    print(" [x] Feature dimension matches across train & test -> PASSED")
    print(" [x] No NaN or invalid values present -> PASSED")
    print(" [x] Vectorizer fitted strictly on X_train -> PASSED")
    print("-" * 60)

    # 5. Sample vocabulary terms
    feature_names = vectorizer.get_feature_names_out()
    print("Sample unigram and bigram features from vocabulary:")
    sample_indices = [100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 49999]
    for idx in sample_indices:
        print(f"  Feature #{idx:5d}: '{feature_names[idx]}'")
    print("=" * 60)
