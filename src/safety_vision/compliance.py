"""PPE-to-person association and temporal compliance decisions."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

class Status(str, Enum):
    COMPLIANT = "compliant"
    MISSING_HELMET = "missing_helmet"
    MISSING_VEST = "missing_vest"
    MISSING_BOTH = "missing_both"
    UNCERTAIN = "uncertain"

@dataclass
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    @property
    def center(self):
        x1,y1,x2,y2=self.box
        return ((x1+x2)/2,(y1+y2)/2)

@dataclass
class PersonObservation:
    person: Detection
    helmet: Detection | None
    vest: Detection | None
    status: Status
    confidence: float

def _inside(person, item, xr, yr):
    px1,py1,px2,py2=person.box; cx,cy=item.center
    return (px1+xr[0]*(px2-px1) <= cx <= px1+xr[1]*(px2-px1)
            and py1+yr[0]*(py2-py1) <= cy <= py1+yr[1]*(py2-py1))

def associate_detections(persons: Iterable[Detection], helmets: Iterable[Detection], vests: Iterable[Detection], cfg: dict) -> list[PersonObservation]:
    a=cfg["association"]; out=[]
    for person in persons:
        hs=[h for h in helmets if _inside(person,h,tuple(a["helmet_center_x"]),tuple(a["helmet_center_y"]))]
        vs=[v for v in vests if _inside(person,v,tuple(a["vest_center_x"]),tuple(a["vest_center_y"]))]
        helmet=max(hs,key=lambda d:d.confidence,default=None); vest=max(vs,key=lambda d:d.confidence,default=None)
        status = Status.COMPLIANT if helmet and vest else Status.MISSING_BOTH if not helmet and not vest else Status.MISSING_HELMET if not helmet else Status.MISSING_VEST
        out.append(PersonObservation(person,helmet,vest,status,min([person.confidence]+([helmet.confidence] if helmet else [])+([vest.confidence] if vest else []))))
    return out

def iou(a,b):
    ax1,ay1,ax2,ay2=a; bx1,by1,bx2,by2=b; ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2)
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); union=(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter
    return inter/union if union else 0.0

@dataclass
class TrackState:
    track_id: int
    box: tuple[float,float,float,float]
    status_history: list[Status] = field(default_factory=list)
    missing_frames: int = 0
    @property
    def status(self):
        if len(self.status_history)<2: return Status.UNCERTAIN
        recent=self.status_history[-2:]
        return recent[-1] if recent[0]==recent[1] else Status.UNCERTAIN

class SimplePersonTracker:
    def __init__(self,cfg):
        t=cfg["tracking"]; self.max_missing=t["max_missing_frames"]; self.match_iou=t["iou_match"]; self.min_confirmed=t["min_confirmed_frames"]; self.next_id=1; self.tracks={}
    def update(self, observations):
        remaining=set(self.tracks); result=[]
        for obs in observations:
            best_id=None; best=self.match_iou
            for tid in remaining:
                score=iou(self.tracks[tid].box,obs.person.box)
                if score>best: best_id,best=tid,score
            if best_id is None: best_id=self.next_id; self.next_id+=1; self.tracks[best_id]=TrackState(best_id,obs.person.box)
            track=self.tracks[best_id]; track.box=obs.person.box; track.missing_frames=0; track.status_history.append(obs.status); track.status_history=track.status_history[-10:]; remaining.discard(best_id)
            status=track.status if len(track.status_history)>=self.min_confirmed else Status.UNCERTAIN; result.append((best_id,obs,status))
        for tid in remaining: self.tracks[tid].missing_frames+=1
        self.tracks={tid:t for tid,t in self.tracks.items() if t.missing_frames<=self.max_missing}
        return result
