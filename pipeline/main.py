import joblib
from pathlib import Path

from pipeline.config import MODELS_DIR
from pipeline.data_loader import load_data_from_postgres
from pipeline.model_training import train_and_evaluate
from pipeline.preprocessing import preprocess_data


def run_pipeline():
    print("Starting MLOps pipeline (retail sales)...")

    print("\n[Step 1] Loading data from PostgreSQL ...")
    df = load_data_from_postgres()
    if df is None:
        print("Pipeline failed at data loading stage.")
        return

    print("\n[Step 2] Preprocessing data...")
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess_data(df)

    joblib.dump(encoders, Path(MODELS_DIR) / "feature_encoders.pkl")
    joblib.dump(scaler, Path(MODELS_DIR) / "scaler.pkl")
    print(f"Saved encoders and scaler to {MODELS_DIR}")

    print("\n[Step 3] Training models...")
    results = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\n" + "=" * 40)
    print("Pipeline execution completed successfully.")
    print("=" * 40)
    for res in results:
        print(
            f"- {res['name']}: MAE={res['mae']:.2f}, "
            f"RMSE={res['rmse']:.2f}, R2={res['r2']:.4f}"
        )
    print("=" * 40)
    print("See metrics.txt for the written report.")


if __name__ == "__main__":
    run_pipeline()
