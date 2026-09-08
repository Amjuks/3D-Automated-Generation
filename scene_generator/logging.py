import json
import sys
import threading
import time


class Log:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()

    def event(self, event, **fields):
        # Callers pass identifiers/counts, never provider payloads or exception bodies.
        row = {"time": time.time(), "event": event, **fields}
        with self.lock:
            with self.path.open("a") as f:
                f.write(json.dumps(row, allow_nan=False) + "\n")
            print(f"[{event}] " + " ".join(f"{k}={v}" for k, v in fields.items()), file=sys.stderr, flush=True)
