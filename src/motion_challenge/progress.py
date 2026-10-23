"""Small, machine-readable progress events; no result or download lives in Git."""
import json
from pathlib import Path
import threading
import time


class Progress:
    def __init__(self, path=None, interval=5.0):
        self.path = Path(path) if path else None
        self.interval = interval
        self.started = time.monotonic()
        self.last = {}
        self.lock = threading.RLock()

    def emit(self, phase, force=False, **values):
        with self.lock:
            now = time.monotonic()
            key = (phase, values.get('file'))
            if not force and now-self.last.get(key, -float('inf')) < self.interval:
                return
            self.last[key] = now
            event = {'phase': phase, 'elapsed_s': round(now-self.started, 3), **values}
            line = json.dumps(event, allow_nan=False)
            print(line, flush=True)
            if self.path:
                temporary = self.path.with_suffix('.tmp')
                temporary.write_text(line+'\n')
                temporary.replace(self.path)
                with self.path.with_suffix('.jsonl').open('a') as stream:
                    stream.write(line+'\n')


def copy_download(response, stream, name, progress=None, total=None):
    progress = progress or Progress()
    total = total or int(response.headers.get('Content-Length', 0)) or None
    count, started = 0, time.monotonic()
    progress.emit('download', force=True, file=str(name), bytes=0, total_bytes=total)
    while block := response.read(1024*1024):
        stream.write(block)
        count += len(block)
        rate = count/max(time.monotonic()-started, 1e-6)
        progress.emit('download', file=str(name), bytes=count, total_bytes=total,
                      bytes_per_s=rate, eta_s=max(0, total-count)/rate if total else None)
    progress.emit('download_received', force=True, file=str(name), bytes=count, total_bytes=total)
    return count
