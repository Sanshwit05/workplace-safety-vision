from pathlib import Path
import argparse
import yaml
from ultralytics import YOLO
from safety_vision.config import load_config
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument("--config",default=ROOT/"config/default.yaml"); p.add_argument("--epochs",type=int); p.add_argument("--device"); a=p.parse_args(); c=load_config(a.config)
data=yaml.safe_load((Path(c["paths"]["dataset"])/"data.yaml").read_text(encoding="utf-8")); data["path"]=c["paths"]["dataset"]; resolved=ROOT/"data"/"resolved_dataset.yaml"; resolved.write_text(yaml.safe_dump(data),encoding="utf-8")
model=YOLO(c["model"]["base"]); model.train(data=str(resolved),epochs=a.epochs or c["model"]["epochs"],imgsz=c["model"]["image_size"],batch=c["model"]["batch"],workers=c["model"]["workers"],device=a.device or c["model"]["device"],amp=c["model"]["amp"],project=str(ROOT/"runs"),name="ppe",exist_ok=True,seed=c["project"]["seed"])
best=ROOT/"runs"/"ppe"/"weights"/"best.pt"; (ROOT/"models").mkdir(exist_ok=True); (ROOT/"models"/"best.pt").write_bytes(best.read_bytes()); print(f"Saved {ROOT/'models/best.pt'}")
