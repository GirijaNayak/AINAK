import json
from ainak.data.validation import validate_jsonl


def test_manifest_validation(tmp_path):
    path=tmp_path/"manifest.jsonl"
    row={"example_id":"r1","recording_id":"r1","timestamp":0,"vision":{"timestamp":0,"feature":[0.]*128},"audio":None,"gesture":None,"priority":.2,"confidence":.8,"alert_type":"informational"}
    path.write_text(json.dumps(row)+"\n",encoding="utf-8")
    count,errors=validate_jsonl(path)
    assert count==1 and not errors


def test_bad_manifest_reports_line(tmp_path):
    path=tmp_path/"bad.jsonl";path.write_text("{}\n",encoding="utf-8")
    count,errors=validate_jsonl(path)
    assert count==0 and "line 1" in errors[0]
