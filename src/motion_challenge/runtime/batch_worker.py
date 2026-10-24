"""Optional adapter batching over private, length-framed JSON connections.

    Model code remains in a worker process. Core slot clients provide only the
    same state/reference packets as the single-environment interface.
"""
import contextlib
from dataclasses import asdict
import json
from multiprocessing.connection import Client, Listener, wait
from pathlib import Path
import sys
import time

from motion_challenge.runtime.encoding import RequestEncoder
from motion_challenge.runtime.worker import load_controller, packet
from motion_challenge.sdk import ControlCommand, ProprioceptiveStatePacket, ReferenceFrame

MAX_MESSAGE = 16 * 1024 * 1024


class BatchClient:
    def __init__(self, address, slot, timeout=120):
        self.timeout = timeout
        self.encoder = RequestEncoder()
        deadline = time.monotonic() + timeout
        while True:
            try:
                self.connection = Client(address, family='AF_UNIX')
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise TimeoutError('batch adapter startup timed out')
                time.sleep(.02)
        self.connection.send_bytes(json.dumps({'slot': slot}).encode())

    def call(self, op, state, reference, seed=0):
        self.connection.send_bytes(self.encoder.request(op, state, reference, seed).encode())
        if not self.connection.poll(self.timeout):
            raise TimeoutError('batch adapter timed out')
        response = json.loads(self.connection.recv_bytes(MAX_MESSAGE))
        if 'error' in response:
            raise RuntimeError(response['error'])
        return ControlCommand(**response['command']) if op == 'act' else None

    def close(self):
        try:
            self.connection.send_bytes(b'{"op":"close"}')
        finally:
            self.connection.close()


def serve(config, address, size):
    runtime = json.loads(Path(config).read_text())
    limit = runtime.get('inference_batch_size', size)
    if type(limit) is not int or limit < 1:
        raise ValueError('inference_batch_size must be a positive integer')
    controller, connections = None, {}
    try:
        with contextlib.redirect_stdout(sys.stderr):
            controller = load_controller(config, 'batch_entrypoint')
        with Listener(address, family='AF_UNIX', backlog=size) as listener:
            listener._listener._socket.settimeout(120)
            for _ in range(size):
                connection = listener.accept()
                if not connection.poll(120):
                    connection.close()
                    raise TimeoutError('batch adapter handshake timed out')
                slot = json.loads(connection.recv_bytes(MAX_MESSAGE))['slot']
                if type(slot) is not int or slot not in range(size) or slot in connections.values():
                    connection.close()
                    raise ValueError('invalid batch adapter slot')
                connections[connection] = slot
            while connections:
                ready = wait(list(connections), timeout=120)
                if not ready:
                    raise TimeoutError('batch adapter clients stopped responding')
                # Coalesce arrivals without waiting for slots that are resetting,
                # running physics, or ending. A global policy barrier can deadlock.
                deadline = time.monotonic() + .0005
                while len(ready) < len(connections):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    more = wait([c for c in connections if c not in ready], timeout=remaining)
                    if not more:
                        break
                    ready.extend(more)
                groups = {'reset': [], 'act': []}
                for connection in ready:
                    request = json.loads(connection.recv_bytes(MAX_MESSAGE))
                    op = request['op']
                    if op == 'close':
                        connection.close()
                        del connections[connection]
                    elif op in groups:
                        groups[op].append((connection, request))
                    else:
                        raise ValueError('unknown batch adapter operation')
                for op, requests in groups.items():
                    for start in range(0, len(requests), limit):
                        chunk = requests[start:start + limit]
                        slots = tuple(connections[c] for c, _ in chunk)
                        states = tuple(packet(ProprioceptiveStatePacket, r['state']) for _, r in chunk)
                        references = tuple(tuple(packet(ReferenceFrame, f) for f in r['reference']) for _, r in chunk)
                        with contextlib.redirect_stdout(sys.stderr):
                            if op == 'reset':
                                controller.reset_batch(slots, states, references, tuple(r['seed'] for _, r in chunk))
                                responses = [{'reset': True} for _ in chunk]
                            else:
                                commands = tuple(controller.act_batch(slots, states, references))
                                if len(commands) != len(chunk) or any(not isinstance(c, ControlCommand) for c in commands):
                                    raise ValueError('batch adapter must return one ControlCommand per slot')
                                responses = [{'command': asdict(c)} for c in commands]
                        for (connection, _), response in zip(chunk, responses):
                            connection.send_bytes(json.dumps(response, allow_nan=False).encode())
    except BaseException as error:
        for connection in connections:
            try:
                connection.send_bytes(json.dumps({'error': f'{type(error).__name__}: {error}'}).encode())
            except (OSError, EOFError):
                pass
        raise
    finally:
        for connection in connections:
            connection.close()
        if controller is not None:
            controller.close()


if __name__ == '__main__':
    serve(sys.argv[1], sys.argv[2], int(sys.argv[3]))
