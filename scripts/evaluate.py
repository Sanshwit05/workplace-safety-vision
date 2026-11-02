from pathlib import Path
import argparse,json
import yaml
from ultralytics import YOLO
from safety_vision.config import load_config
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument("--config",default=ROOT/"config/default.yaml"); a=p.parse_args(); c=load_config(a.config)
data=yaml.safe_load((Path(c["paths"]["dataset"])/"data.yaml").read_text(encoding="utf-8")); data["path"]=c["paths"]["dataset"]; resolved=ROOT/"data"/"resolved_dataset.yaml"; resolved.write_text(yaml.safe_dump(data),encoding="utf-8")
model=YOLO(c["paths"]["model"]); m=model.val(data=str(resolved),split="test",imgsz=c["model"]["image_size"],batch=c["model"]["batch"],workers=c["model"]["workers"],device=c["model"]["device"],plots=True,project=str(ROOT/"reports"),name="evaluation",exist_ok=True)
out={"precision":float(m.box.mp),"recall":float(m.box.mr),"mAP50":float(m.box.map50),"mAP50-95":float(m.box.map),"confusion_matrix":str(ROOT/"reports/evaluation/confusion_matrix.png")}; (ROOT/"reports").mkdir(exist_ok=True); (ROOT/"reports"/"metrics.json").write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
