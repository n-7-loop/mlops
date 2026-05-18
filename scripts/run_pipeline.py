"""
Entry point: run from project root as
  python scripts/run_pipeline.py
or
  python -m scripts.run_pipeline
(from project root, with PYTHONPATH including the project root)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.main import run_pipeline  # noqa: E402

if __name__ == "__main__":
    run_pipeline()
