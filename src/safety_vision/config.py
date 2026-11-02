from pathlib import Path
from typing import Any
import yaml

def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(path).resolve().parents[1]
    for key in ("dataset", "model", "outputs", "reports"):
        cfg["paths"][key] = str((root / cfg["paths"][key]).resolve())
    return cfg
