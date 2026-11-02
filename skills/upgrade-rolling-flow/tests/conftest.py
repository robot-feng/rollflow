from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for d in (ROOT / 'assets', ROOT / 'tools'):
    sys.path.insert(0, str(d))
