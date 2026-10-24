"""Bounded asynchronous gzip blocks: no open file descriptor per active motion."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import threading


class TraceWriter:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=1,thread_name_prefix='trace-writer')
        self.capacity = threading.Semaphore(16)
        self.error = None

    def submit(self, path, text):
        self.capacity.acquire()
        def write():
            try:
                if self.error is not None:
                    raise self.error
                with path.open('ab') as stream:
                    stream.write(gzip.compress(text.encode(),compresslevel=1,mtime=0))
            except BaseException as error:
                self.error = error
                raise
            finally:
                self.capacity.release()
        return self.pool.submit(write)

    def open(self, path):
        path.write_bytes(b'')
        return Trace(self,path)

    def close(self):
        self.pool.shutdown(wait=True)
        if self.error is not None:
            raise self.error


class Trace:
    def __init__(self, writer, path):
        self.writer,self.path = writer,path
        self.parts,self.size,self.last = [],0,None

    def write(self, text):
        self.parts.append(text)
        self.size += len(text)
        if self.size >= 16384:
            self.flush()

    def flush(self):
        if self.parts:
            self.last = self.writer.submit(self.path,''.join(self.parts))
            self.parts,self.size = [],0

    def close(self):
        self.flush()
        if self.last is not None:
            self.last.result()
        if self.writer.error is not None:
            raise self.writer.error
