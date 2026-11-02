"""
Dataset ingestion, validation, and stratified splitting module.
Handles loading raw movie reviews, data quality checks, and train/test splitting.
"""

from pathlib import Path
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_PATH, RANDOM_STATE, TEST_SIZE

REQUIRED_COLUMNS = ["review", "sentiment"]


def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Loads the movie review dataset from CSV and verifies required columns.

    Args:
        data_path: Path to the dataset CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If required columns ('review', 'sentiment') are missing.
    """
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {data_path.resolve()}. "
            "Please ensure the file is placed in the data/ directory."
        )

    df = pd.read_csv(data_path)

    # Validate column structure
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return df


def inspect_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates summary statistics on missing values, duplicates, and class distributions.

    Args:
        df: Input DataFrame.

    Returns:
        dict: Inspection metrics.
    """
    metrics = {
        "shape": df.shape,
        "missing_values": df[REQUIRED_COLUMNS].isnull().sum().to_dict(),
        "duplicate_count": int(df.duplicated(subset=REQUIRED_COLUMNS).sum()),
        "class_distribution": df["sentiment"].value_counts().to_dict(),
        "class_proportions": df["sentiment"].value_counts(normalize=True).to_dict(),
    }
    return metrics


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Applies initial data quality cleaning:
    - Drops rows with missing reviews or sentiment labels
    - Removes exact duplicate review rows to prevent test snooping / leakage
    - Resets dataframe index

    Args:
        df: Raw DataFrame.
        drop_duplicates: Whether to drop duplicate review entries.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    initial_count = len(df)

    # 1. Handle missing values
    df_cleaned = df.dropna(subset=REQUIRED_COLUMNS).copy()
    dropped_na = initial_count - len(df_cleaned)

    # 2. Remove duplicate rows
    if drop_duplicates:
        df_cleaned = df_cleaned.drop_duplicates(subset=REQUIRED_COLUMNS, keep="first").copy()
    dropped_duplicates = (initial_count - dropped_na) - len(df_cleaned)

    df_cleaned = df_cleaned.reset_index(drop=True)
    return df_cleaned


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Performs a stratified train/test split on raw reviews and sentiment labels.
    Using stratify=y preserves the exact class balance in both partitions.

    Args:
        df: Cleaned DataFrame containing 'review' and 'sentiment' columns.
        test_size: Ratio of data allocated to the test set (default 0.2).
        random_state: Seed for reproducibility (default 42).

    Returns:
        X_train, X_test, y_train, y_test: Raw review series and label series.
    """
    X = df["review"]
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def load_and_split_data(
    data_path: Path = DATA_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    High-level convenience function for model training:
    Loads raw data, performs quality cleaning, and returns stratified splits.

    Returns:
        X_train, X_test, y_train, y_test
    """
    df_raw = load_raw_data(data_path)
    df_clean = clean_dataset(df_raw, drop_duplicates=True)
    return split_data(df_clean, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 2: DATASET INGESTION & INSPECTION REPORT")
    print("=" * 60)

    # 1. Load data
    raw_df = load_raw_data()
    raw_inspection = inspect_data(raw_df)

    print(f"Original shape: {raw_inspection['shape']}")
    print(f"Missing values: {raw_inspection['missing_values']}")
    print(f"Duplicate count before cleaning: {raw_inspection['duplicate_count']}")
    print(f"Class distribution: {raw_inspection['class_distribution']}")

    # 2. Clean data
    cleaned_df = clean_dataset(raw_df, drop_duplicates=True)
    clean_inspection = inspect_data(cleaned_df)
    print(f"Shape after cleaning: {clean_inspection['shape']}")

    # 3. Stratified Train / Test Split
    X_train, X_test, y_train, y_test = split_data(cleaned_df)

    print("-" * 60)
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    print(f"Training class distribution:\n{y_train.value_counts().to_dict()}")
    print(f"Test class distribution:\n{y_test.value_counts().to_dict()}")

    # 4. Proportions check
    train_pos_ratio = (y_train == "positive").mean()
    test_pos_ratio = (y_test == "positive").mean()
    print("-" * 60)
    print(f"Training positive ratio: {train_pos_ratio:.4f} ({train_pos_ratio*100:.2f}%)")
    print(f"Test positive ratio:     {test_pos_ratio:.4f} ({test_pos_ratio*100:.2f}%)")
    print(f"Proportion preserved:   {abs(train_pos_ratio - test_pos_ratio) < 0.001}")
    print("=" * 60)
