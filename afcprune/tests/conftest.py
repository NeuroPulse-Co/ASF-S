import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # afcprune/ (the project dir holding the package)
sys.path.insert(0, str(ROOT))
