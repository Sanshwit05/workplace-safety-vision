"""Create a local verification video from real held-out dataset frames."""
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[1]; images=sorted((ROOT/"data"/"construction-ppe"/"images"/"test").glob("*.jpg"))[:30]; out=ROOT/"outputs"/"dataset_sample.mp4"; out.parent.mkdir(exist_ok=True)
first=cv2.imread(str(images[0])); writer=cv2.VideoWriter(str(out),cv2.VideoWriter_fourcc(*"mp4v"),10,(first.shape[1],first.shape[0]))
for path in images:
    frame=cv2.imread(str(path)); writer.write(frame)
writer.release(); print(out)
