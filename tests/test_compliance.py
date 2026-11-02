from safety_vision.compliance import *
CFG={"association":{"helmet_center_x":[.15,.85],"helmet_center_y":[-.05,.55],"vest_center_x":[.1,.9],"vest_center_y":[.2,.95]},"tracking":{"max_missing_frames":2,"iou_match":.25,"min_confirmed_frames":3}}
def d(label,box): return Detection(label,.9,box)
def test_association_states():
    p=[d("person",(0,0,100,200))]; h=[d("helmet",(30,10,70,40))]; v=[d("vest",(20,70,80,160))]
    assert associate_detections(p,h,v,CFG)[0].status==Status.COMPLIANT
    assert associate_detections(p,[],v,CFG)[0].status==Status.MISSING_HELMET
    assert associate_detections(p,[],[],CFG)[0].status==Status.MISSING_BOTH
def test_persistence_suppresses_single_frame_error():
    t=SimplePersonTracker(CFG); p=d("person",(0,0,100,200)); h=d("helmet",(30,10,70,40)); v=d("vest",(20,70,80,160))
    assert t.update(associate_detections([p],[h],[v],CFG))[0][2]==Status.UNCERTAIN
    t.update(associate_detections([p],[],[v],CFG)); t.update(associate_detections([p],[h],[v],CFG)); assert t.update(associate_detections([p],[h],[v],CFG))[0][2]==Status.COMPLIANT
