import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".runtime-cache"
MPL_CACHE = CACHE_ROOT / "matplotlib"

MPL_CACHE.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
