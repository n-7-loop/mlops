import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Feature order must match training when building the inference vector.
CATEGORICAL_COLS = [
    "branch",
    "city",
    "customer_type",
    "gender",
    "product_name",
    "product_category",
]
NUMERIC_COLS = ["unit_price", "quantity"]
TARGET_COL = "total_price"


def preprocess_data(df):
    """
    Cleans sales data, encodes categoricals, scales features, and splits train/test.
    Predicts total sale amount from basket and customer context (no tax in features).
    """
    df = df.copy()
    df.columns = [col.lower() for col in df.columns]

    required = CATEGORICAL_COLS + NUMERIC_COLS + [TARGET_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].dropna().copy()

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str).str.strip())
        encoders[col] = le

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X = df[feature_cols].astype(float)
    y = pd.to_numeric(df[TARGET_COL], errors="coerce")
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"Data preprocessed. Train size: {len(X_train)}, Test size: {len(X_test)}")

    return X_train, X_test, y_train, y_test, encoders, scaler
