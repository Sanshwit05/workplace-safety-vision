from pathlib import Path
import argparse,cv2,json
from ultralytics import YOLO
from safety_vision.config import load_config
from safety_vision.compliance import Detection,associate_detections,SimplePersonTracker
from safety_vision.reporting import write_reports
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(); p.add_argument("source"); p.add_argument("--config",default=ROOT/"config/default.yaml"); p.add_argument("--output"); p.add_argument("--model"); a=p.parse_args(); c=load_config(a.config)
model=YOLO(a.model or c["paths"]["model"]); names=model.names; wanted={k:{x.lower() for x in v} for k,v in c["classes"].items()}; tracker=SimplePersonTracker(c); records=[]; src=Path(a.source); video=src.suffix.lower() in {".mp4",".avi",".mov",".mkv"}; out=Path(a.output or (ROOT/"outputs"/(src.stem+("_annotated.mp4" if video else "_annotated.jpg")))); out.parent.mkdir(parents=True,exist_ok=True); cap=cv2.VideoCapture(str(src)) if video else None; writer=None; frame_no=0
while True:
    if video: ok,frame=cap.read()
    else: frame=cv2.imread(str(src)); ok=frame is not None
    if not ok: break
    if frame_no % int(c["video"]["sample_every"]) != 0:
        if video:
            if writer is None: writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),c["video"]["output_fps"] or cap.get(cv2.CAP_PROP_FPS) or 25,(frame.shape[1],frame.shape[0]))
            writer.write(frame)
        frame_no+=1
        continue
    result=model.predict(frame,conf=c["model"]["conf"],iou=c["model"]["iou"],device=c["model"]["device"],verbose=False,max_det=c["model"]["max_det"])[0]; dets=[]
    for box,conf,cls in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist(),result.boxes.cls.cpu().tolist()): dets.append(Detection(str(names[int(cls)]).lower(),float(conf),tuple(map(float,box))))
    persons=[d for d in dets if d.label in wanted["person"]]; helmets=[d for d in dets if d.label in wanted["helmet"]]; vests=[d for d in dets if d.label in wanted["vest"]]
    for tid,obs,status in tracker.update(associate_detections(persons,helmets,vests,c)):
        x1,y1,x2,y2=map(int,obs.person.box); color=(0,180,0) if status.value=="compliant" else ((0,165,255) if status.value=="uncertain" else (0,0,255)); cv2.rectangle(frame,(x1,y1),(x2,y2),color,2); cv2.putText(frame,f"ID {tid}: {status.value}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2); records.append({"source":str(src),"frame":frame_no,"track_id":tid,"status":status.value,"confidence":round(obs.confidence,4),"x1":x1,"y1":y1,"x2":x2,"y2":y2})
    if video:
        if writer is None: writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),c["video"]["output_fps"] or cap.get(cv2.CAP_PROP_FPS) or 25,(frame.shape[1],frame.shape[0]))
        writer.write(frame)
    else: cv2.imwrite(str(out),frame); break
    frame_no+=1
if cap: cap.release()
if writer: writer.release()
print(json.dumps({"output":str(out),"report":write_reports(records,c["paths"]["reports"],src.stem)},indent=2))
