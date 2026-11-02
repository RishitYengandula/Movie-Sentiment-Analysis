"""
Configuration module for the Movie Review Sentiment Analysis project.
Defines central filesystem paths, random seeds, and pipeline constants.
"""

from pathlib import Path

# Project root directory (resolves to the repository root: Sentiment_Analysis/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directory and default dataset file path
DATA_DIR = PROJECT_ROOT / "data"
DATA_PATH = DATA_DIR / "IMDB Dataset.csv"

# Model artifacts directory and default pipeline artifact path
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"

# Reports directory for evaluation charts and metrics
REPORTS_DIR = PROJECT_ROOT / "reports"

# Pipeline constants & reproducibility settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

