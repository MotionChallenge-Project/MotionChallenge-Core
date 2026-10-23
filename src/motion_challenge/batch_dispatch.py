"""Independent evaluation processes feeding one official MuJoCo CPU batch.

Only small PD commands and generalized states cross the physics connection.
Reference FK, policy JSON, history and scoring stay parallel in slot processes.
Connections are private core-to-core IPC, not an untrusted adapter interface.
"""
import json
import math
from multiprocessing.connection import Client, Listener, wait
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

from .batch_physics import BatchPool, Request
from .physics import World, command_values, forward_kinematics
from .progress import Progress


class RemoteWorld(World):
    def __init__(self, profile, address, slot):
        super().__init__(profile)
        self.connection = Client(address, family='AF_UNIX')
        self.connection.send(slot)

    def _receive(self):
        reply = self.connection.recv()
        if isinstance(reply, dict):
            raise RuntimeError(reply['error'])
        self.data.time, qpos, qvel = reply
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        # State packets need body-origin FK/Jacobians, not an extra constraint
        # solve. The authoritative persistent solver state stays in Batch.
        forward_kinematics(self.model, self.data)

    def reset(self, motion):
        super().reset(motion)
        self.connection.send(('reset', self.data.qpos.copy(), self.data.qvel.copy()))
        self._receive()
        return self.state()

    def advance(self, command, until_s):
        self.connection.send(('advance', command_values(command, len(self.joints)), until_s))
        self._receive()

    def close(self):
        self.connection.send(('done',))
        self.connection.close()


def evaluate_batch(args, selected, results, protocol, full_suite):
    size = min(args.batch_size, len(selected))
    progress = Progress(args.output.with_suffix('.progress.json'), args.progress_interval)
    pool = BatchPool(args.robot, size, args.threads)
    results['batch_size'] = size
    results['batch_threads'] = pool.batch.nthread
    results['execution'] = 'multiprocess_reference_policy_scoring_shared_mujoco_batch'
    records, processes, connections = [], [], {}
    log_threads = []
    states = {}
    def log_output(slot, process, path):
        with path.open('w') as log:
            for line in process.stdout:
                log.write(line)
                log.flush()
                try:
                    event = json.loads(line)
                except ValueError:
                    print(line, end='', flush=True)
                    continue
                if event.get('phase') == 'evaluation':
                    with progress.lock:
                        old = states.get(slot, {}).get('completed', 0)
                        states[slot] = event
                        active = {str(i): next(iter(e.get('active', {}).values()))
                                  for i, e in states.items() if e.get('active')}
                        progress.emit('evaluation', force=event['completed'] != old,
                            completed=sum(e['completed'] for e in states.values()), total=len(selected),
                            passed=sum(e['passed'] for e in states.values()), active=active, backend='batch')
    try:
        # Keep the UNIX socket path short and canonical on macOS.
        socket_root = '/tmp' if sys.platform == 'darwin' else None
        with tempfile.TemporaryDirectory(prefix='mc-batch-', dir=socket_root) as directory:
            address = str(Path(directory)/'physics.sock')
            with Listener(address, family='AF_UNIX') as listener:
                # A crashed child must not leave the coordinator in accept().
                listener._listener._socket.settimeout(120)
                for slot in range(size):
                    path = args.output.with_name(args.output.stem+f'.part{slot:02d}.json')
                    records.append(path)
                    command = [sys.executable, '-m', 'motion_challenge.run', '--backend', 'batch-client',
                               '--batch-address', address, '--batch-slot', str(slot), '--output', str(path),
                               '--progress-interval', str(args.progress_interval)]
                    for name in ('controller', 'suite', 'protocol', 'robot', 'data'):
                        command += ['--'+name, str(getattr(args, name))]
                    for clip in selected[slot::size]:
                        command += ['--clip-id', clip['path']]
                    if args.max_duration is not None:
                        command += ['--max-duration', str(args.max_duration)]
                    if args.worker_python:
                        command += ['--worker-python', args.worker_python]
                    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                               text=True, bufsize=1,
                                               env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
                    processes.append(process)
                    thread = threading.Thread(target=log_output, args=(slot, process, path.with_suffix('.log')), daemon=True)
                    log_threads.append(thread)
                    thread.start()
                for _ in range(size):
                    connection = listener.accept()
                    slot = connection.recv()
                    if slot not in range(size) or slot in connections.values():
                        connection.close()
                        raise ValueError('invalid batch slot handshake')
                    connections[connection] = slot
                pending = {}
                while connections:
                    ready = wait([c for c in connections if connections[c] not in pending], timeout=1)
                    if not ready and any(p.poll() not in (None, 0) for p in processes):
                        raise RuntimeError('batch slot process failed; see part logs')
                    for connection in ready:
                        slot = connections[connection]
                        request = connection.recv()
                        if request[0] == 'done':
                            connection.close()
                            del connections[connection]
                        elif request[0] == 'reset':
                            pool.worlds[slot].data.qpos[:] = request[1]
                            pool.worlds[slot].data.qvel[:] = request[2]
                            pending[slot] = Request('reset')
                        elif request[0] == 'advance':
                            pending[slot] = Request('advance', request[1], request[2])
                        else:
                            raise ValueError('unknown batch physics request')
                    if pending and set(connections.values()) <= pending.keys():
                        pool._execute(pending)
                        for connection, slot in connections.items():
                            connection.send((float(pool.time[slot]), pool.qpos[slot], pool.qvel[slot]))
                        pending.clear()
                for process in processes:
                    if process.wait(timeout=30):
                        raise RuntimeError('batch slot failed after simulation')
        for thread in log_threads:
            thread.join(timeout=5)
        parts = [json.loads(path.read_text()) for path in records]
        if any(p['status'] != 'completed' for p in parts):
            raise ValueError('incomplete batch part')
        clips = {c['id']: c for part in parts for c in part['clips']}
        if len(clips) != sum(len(p['clips']) for p in parts) or set(clips) != {c['path'] for c in selected}:
            raise ValueError('batch output has duplicate or missing clips')
        results['clips'] = [clips[c['path']] for c in selected]
        results.update(status='completed', full_suite=full_suite,
            success_count=sum(c['success'] for c in results['clips']),
            macro_completion=math.fsum(c['completion'] for c in results['clips'])/len(selected))
        if protocol.get('position_alignment_metrics'):
            results['macro_metrics_over_observed_prefix'] = {
                k: math.fsum(c['metric_means_over_observed_prefix'][k] for c in results['clips'])/len(selected)
                for k in results['clips'][0]['metric_means_over_observed_prefix']}
    except BaseException as error:
        results.update(status='infrastructure_error', error=f'{type(error).__name__}: {error}')
        for connection in connections:
            connection.close()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        args.output.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')
