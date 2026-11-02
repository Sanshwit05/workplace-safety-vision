from safety_vision.reporting import summarize,write_reports
def test_summary_and_files(tmp_path):
    rows=[{"source":"x","frame":0,"track_id":1,"status":"compliant","confidence":.9,"x1":0,"y1":0,"x2":1,"y2":1},{"source":"x","frame":1,"track_id":1,"status":"missing_helmet","confidence":.8,"x1":0,"y1":0,"x2":1,"y2":1}]
    assert summarize(rows)["compliance_rate"]==.5; paths=write_reports(rows,tmp_path); assert (tmp_path/"violations.csv").exists() and (tmp_path/"violations.json").exists()
