import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SUNFLOWER_ROOT", str(ROOT))
