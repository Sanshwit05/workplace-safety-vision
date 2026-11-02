from pathlib import Path
import argparse
import yaml
from ultralytics import YOLO
from safety_vision.config import load_config
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument("--config",default=ROOT/"config/default.yaml"); p.add_argument("--split",default="val"); a=p.parse_args(); c=load_config(a.config)
data=yaml.safe_load((Path(c["paths"]["dataset"])/"data.yaml").read_text(encoding="utf-8")); data["path"]=c["paths"]["dataset"]; resolved=ROOT/"data"/"resolved_dataset.yaml"; resolved.write_text(yaml.safe_dump(data),encoding="utf-8")
model=YOLO(c["paths"]["model"] if Path(c["paths"]["model"]).exists() else c["model"]["base"]); m=model.val(data=str(resolved),split=a.split,imgsz=c["model"]["image_size"],batch=c["model"]["batch"],workers=c["model"]["workers"],device=c["model"]["device"],project=str(ROOT/"runs"),name="validation",exist_ok=True); print({"precision":m.box.mp,"recall":m.box.mr,"mAP50":m.box.map50,"mAP50-95":m.box.map})
