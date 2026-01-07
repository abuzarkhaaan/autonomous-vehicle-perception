import json, time
from pathlib import Path

class Logger:
    def __init__(self, log_dir="outputs/logs", run_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name or time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = self.log_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"

    def log(self, **kwargs):
        payload = {"t": time.time(), **kwargs}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return payload

    def info(self, msg, **kwargs):
        return self.log(level="info", msg=msg, **kwargs)

    def warn(self, msg, **kwargs):
        return self.log(level="warn", msg=msg, **kwargs)

    def error(self, msg, **kwargs):
        return self.log(level="error", msg=msg, **kwargs)
