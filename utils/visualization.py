import json
from pathlib import Path

def save_json(path, obj):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def make_stub_bev_artifact(predictions, meta=None):
    return {"type": "bev_stub", "meta": meta or {}, "predictions": predictions}
