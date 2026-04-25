import os
import warnings
from pathlib import Path


def configure_runtime_environment() -> None:
    root = Path(__file__).resolve().parent.parent
    cache_root = root / ".runtime-cache"
    mpl_cache = cache_root / "matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("GLOG_minloglevel", "2")
    warnings.filterwarnings(
        "ignore",
        message="SymbolDatabase.GetPrototype\\(\\) is deprecated.*",
    )
