from pathlib import Path
import urllib.request, zipfile
ROOT=Path(__file__).resolve().parents[1]; URL="https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"; dest=ROOT/"data"/"construction-ppe"; archive=ROOT/"data"/"construction-ppe.zip"; dest.parent.mkdir(parents=True,exist_ok=True)
if not dest.exists() or not any(dest.iterdir()):
    print(f"Downloading {URL}"); urllib.request.urlretrieve(URL,archive)
    with zipfile.ZipFile(archive) as zf: zf.extractall(dest.parent)
# The archive contains data.yaml/images/labels at its root; normalize that
# layout under data/construction-ppe so every project artifact stays in scope.
if (dest.parent/"images").exists() and not (dest/"images").exists():
    for name in ("images", "labels", "data.yaml", "LICENSE"):
        src=dest.parent/name
        if src.exists(): src.replace(dest/name)
print(f"Dataset ready: {dest}")
