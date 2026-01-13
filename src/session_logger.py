import json
import time

class LiveSessionLogger:
    def __init__(self):
        self.records = []
        self.start_time = time.time()

    def log(self, data: dict):
        data["t"] = time.time() - self.start_time
        self.records.append(data)

    def dump(self):
        ts = int(time.time())
        fname = f"session_{ts}.json"
        with open(fname, "w") as f:
            json.dump(self.records, f, indent=2)
        return fname
