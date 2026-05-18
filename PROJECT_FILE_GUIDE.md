# MLOps Assignment - File-by-File Guide

This document explains what each important file in this project does and how they connect in the full workflow.

---

## 1) Root-Level Files

### `README.md`
- Project setup and run instructions.
- Explains PostgreSQL requirement and default DB URL.
- Shows the expected run order:
  1. Install dependencies
  2. Run training pipeline (`python scripts/run_pipeline.py`)
  3. Start app (`python app.py`)
- Mentions first-run DB seeding behavior from CSV.

### `requirements.txt`
- Declares Python dependencies:
  - Flask for web app/API
  - pandas and numpy for data work
  - scikit-learn and joblib for ML + artifact persistence
  - psycopg for PostgreSQL access

### `app.py`
- Main Flask inference service.
- Loads trained artifacts at startup:
  - `models/RandomForest.pkl`
  - `models/scaler.pkl`
  - `models/feature_encoders.pkl`
- Validates artifact compatibility with expected feature schema.
- Exposes 3 routes:
  - `/` -> serves UI from `templates/index.html`
  - `/api/form-options` -> returns allowed categorical values from saved encoders
  - `/predict` -> accepts JSON input, encodes/scales features, returns predicted total
- Includes safe error responses for missing artifacts, unknown categories, and invalid input.

### `metrics.txt`
- Generated output file from training.
- Stores evaluation metrics for each trained model:
  - MAE
  - RMSE
  - R2 Score
- Useful for model comparison in demo and model selection decisions.

### `sales.csv`
- Root-level copy of dataset.
- Used as fallback source if `data/sales.csv` is missing.
- Copied into `data/sales.csv` by loader logic on first run if needed.

---

## 2) Pipeline Package (`pipeline/`)

### `pipeline/__init__.py`
- Marks `pipeline` as a Python package.
- No core logic; used for imports like `from pipeline.main import run_pipeline`.

### `pipeline/config.py`
- Centralized configuration module.
- Defines:
  - `ROOT_DIR`
  - `DATA_DIR`, `DATA_FILE`
  - `SALES_TABLE`
  - `DATABASE_URL` (reads from env var, otherwise uses default local URL)
  - `MODELS_DIR`
  - `METRICS_FILE`
- Ensures required directories exist (`models/`, `data/`).
- This file is the single source of truth for paths and DB connection config.

### `pipeline/data_loader.py`
- Handles data ingestion from PostgreSQL and one-time table seeding.
- Core functions:
  - `_ensure_sales_csv()`
    - If `data/sales.csv` is missing but root `sales.csv` exists, copy it.
  - `load_data_from_csv()`
    - Read CSV into pandas DataFrame (for seeding path).
  - `_ensure_sales_table(conn)`
    - Creates `sales` table if it does not exist.
  - `_seed_table_from_csv_if_empty(conn)`
    - Checks row count; if empty, inserts rows from CSV.
    - Uses `ON CONFLICT (sale_id) DO NOTHING` to avoid duplicate key crashes.
  - `load_data_from_postgres()`
    - Main pipeline data source.
    - Ensures schema, seeds if needed, then fetches all rows ordered by `sale_id`.
- Purpose: move from file-based training toward persistent DB-backed training data.

### `pipeline/preprocessing.py`
- Converts raw DataFrame into train/test ML-ready matrices.
- Defines modeling schema:
  - Categorical features: `branch`, `city`, `customer_type`, `gender`, `product_name`, `product_category`
  - Numeric features: `unit_price`, `quantity`
  - Target: `total_price`
- Steps performed:
  1. Normalize column names to lowercase
  2. Validate required columns exist
  3. Drop rows with null values in required columns
  4. Label-encode categorical columns
  5. Build feature matrix and numeric target
  6. Scale features with `StandardScaler`
  7. Split train/test (80/20, fixed random state)
- Returns train/test arrays plus fitted encoders and scaler.

### `pipeline/model_training.py`
- Trains and evaluates multiple regression models.
- Models trained:
  - `LinearRegression`
  - `RandomForestRegressor`
  - `SVR`
- For each model:
  - Fit on training set
  - Predict on test set
  - Compute MAE, RMSE, R2
  - Append metrics to `metrics.txt`
  - Save model artifact to `models/<ModelName>.pkl`
- Returns structured metric summaries used by pipeline logger output.

### `pipeline/main.py`
- Pipeline orchestration entrypoint logic.
- End-to-end flow:
  1. Load dataset from PostgreSQL (`load_data_from_postgres`)
  2. Preprocess (`preprocess_data`)
  3. Save preprocessing artifacts:
     - `feature_encoders.pkl`
     - `scaler.pkl`
  4. Train models + evaluate (`train_and_evaluate`)
  5. Print final run summary
- This is the central "run everything" function for ML training side.

---

## 3) Scripts Package (`scripts/`)

### `scripts/run_pipeline.py`
- Script-friendly launcher for training pipeline.
- Adds project root to `sys.path` so imports work when executed directly.
- Calls `run_pipeline()` from `pipeline.main`.
- Primary command used in docs: `python scripts/run_pipeline.py`.

### `scripts/__init__.py`
- Marks `scripts` as a package.
- No business logic.

---

## 4) Frontend Template

### `templates/index.html`
- Client-side UI for entering features and requesting predictions.
- Includes:
  - Styled form layout and UX states (loading, error, success)
  - Dropdown fields for categorical columns
  - Numeric inputs for `unit_price` and `quantity`
- JavaScript behavior:
  - On load: call `/api/form-options`, populate dropdowns from saved encoders
  - On submit: call `/predict` with JSON payload
  - Display predicted value or API error message
- Keeps UI constrained to values the model has seen in training for categories.

---

## 5) Data and Artifacts Directories

### `data/sales.csv`
- Canonical training seed dataset location expected by pipeline.
- Used to populate PostgreSQL table only when table is empty.

### `models/`
- Stores generated binary artifacts after running pipeline:
  - `RandomForest.pkl`
  - `LinearRegression.pkl`
  - `SVR.pkl`
  - `scaler.pkl`
  - `feature_encoders.pkl`
- Inference app reads these files at startup.

### `pipeline/__pycache__/` and `__pycache__/`
- Auto-generated Python bytecode cache files (`.pyc`).
- Not handwritten source code and not part of core logic explanation.

---

## 6) How Everything Connects (End-to-End)

1. Run `python scripts/run_pipeline.py`
2. Pipeline ensures PostgreSQL table exists and seeds from CSV if empty.
3. Pipeline preprocesses data and trains multiple models.
4. Pipeline writes metrics and saves model/scaler/encoder artifacts.
5. Run `python app.py`
6. Flask loads artifacts and serves web UI/API.
7. UI asks API for valid dropdown options and sends prediction requests.
8. API returns predicted `total_price` from trained `RandomForest` model.

This is the complete training + serving loop implemented in this repository.
