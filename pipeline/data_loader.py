import shutil
import pandas as pd
from sqlalchemy import create_engine

from pipeline.config import (
    DATA_DIR,
    DATA_FILE,
    DATABASE_URL,
    ROOT_DIR,
    SALES_TABLE,
)


def _ensure_sales_csv():
    """
    If data/sales.csv is missing but sales.csv exists at project root,
    copy it into data/ folder.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    root_csv = ROOT_DIR / "sales.csv"

    if not DATA_FILE.exists() and root_csv.exists():
        shutil.copy2(root_csv, DATA_FILE)
        print(f"Copied '{root_csv.name}' -> '{DATA_FILE}'")


def load_data_from_csv():
    """
    Load dataset from CSV.
    """
    _ensure_sales_csv()

    try:
        df = pd.read_csv(DATA_FILE)
        print(f"Successfully loaded {len(df)} rows from CSV")
        return df

    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


def load_data_from_postgres():
    """
    Load dataset into MySQL using XAMPP.
    """

    try:
        # Create MySQL connection
        engine = create_engine(DATABASE_URL)

        # Load CSV
        df = load_data_from_csv()

        if df is None or df.empty:
            raise ValueError("CSV data not found.")

        # Save to MySQL
        df.to_sql(SALES_TABLE, engine, if_exists="replace", index=False)

        print(f"Inserted {len(df)} rows into MySQL table '{SALES_TABLE}'")

        # Read data back from MySQL
        df = pd.read_sql(f"SELECT * FROM {SALES_TABLE}", engine)

        print(f"Successfully loaded {len(df)} rows from MySQL")

        return df

    except Exception as e:
        print(f"Error loading data from MySQL: {e}")
        return None


if __name__ == "__main__":
    data = load_data_from_postgres()

    if data is not None:
        print(data.head())