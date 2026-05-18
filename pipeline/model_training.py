import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pipeline.config import MODELS_DIR, METRICS_FILE


def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains 3 regressors on scaled features, saves models, and writes metrics.
    """
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "SVR": SVR(),
    }

    results = []

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        f.write("Retail sales total_price regression - model performance\n")
        f.write("=" * 50 + "\n\n")

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)

        print(f"{name} -> MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.4f}")

        with open(METRICS_FILE, "a", encoding="utf-8") as f:
            f.write(f"Model: {name}\n")
            f.write(f"MAE: {mae:.4f}\n")
            f.write(f"RMSE: {rmse:.4f}\n")
            f.write(f"R2 Score: {r2:.4f}\n")
            f.write("-" * 40 + "\n\n")

        model_path = Path(MODELS_DIR) / f"{name}.pkl"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")

        results.append({"name": name, "mae": mae, "rmse": rmse, "r2": r2, "path": model_path})

    return results
