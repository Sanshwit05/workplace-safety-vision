import csv, json
from collections import Counter
from pathlib import Path

def summarize(records):
    counts=Counter(r["status"] for r in records); total=len(records); compliant=counts.get("compliant",0)
    return {"observations":total,"compliant":compliant,"violations":total-compliant,"compliance_rate":round(compliant/total,4) if total else 0.0,"status_counts":dict(counts)}

def write_reports(records,output_dir,stem="violations"):
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True); csv_path=output_dir/f"{stem}.csv"; json_path=output_dir/f"{stem}.json"; fields=["source","frame","track_id","status","confidence","x1","y1","x2","y2"]
    with csv_path.open("w",newline="",encoding="utf-8") as f: writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader(); writer.writerows(records)
    payload={"summary":summarize(records),"records":records}; json_path.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    return {"csv":str(csv_path),"json":str(json_path),"summary":payload["summary"]}
