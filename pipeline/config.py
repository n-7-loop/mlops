import os
from pathlib import Path

# Project root (parent of the `pipeline` package directory)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Data
DATA_DIR = ROOT_DIR / "data"
DATA_FILE = DATA_DIR / "sales.csv"
SALES_TABLE = "sales"

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:@localhost/mlopsassignment",
)

# Artifacts
MODELS_DIR = ROOT_DIR / "models"
METRICS_FILE = ROOT_DIR / "metrics.txt"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)