"""JSON-lines worker entrypoint. Process separation, not a hostile-code sandbox."""
import contextlib
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import sys
import traceback

from .sdk import BodyState, ReferenceFrame, StatePacket


def packet(cls, data):
    data = dict(data)
    data['bodies'] = tuple(BodyState(**b) for b in data.get('bodies', []))
    if 'joint_names' in data:
        data['joint_names'] = tuple(data['joint_names'])
    return cls(**data)


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    root = Path(cfg.get('adapter_root', Path(sys.argv[1]).resolve().parent)).resolve()
    sys.path.insert(0, str(root))
    for path in cfg.get('python_paths', []):
        sys.path.insert(0, str((root/path).resolve()))
    controller = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            module, factory = cfg['entrypoint'].split(':')
            controller = getattr(importlib.import_module(module), factory)(root, cfg)
        print(json.dumps({'ready': True}), flush=True)
        for line in sys.stdin:
            request = json.loads(line)
            if request['op'] == 'close':
                break
            state = packet(StatePacket, request['state'])
            refs = tuple(packet(ReferenceFrame, f) for f in request['reference'])
            with contextlib.redirect_stdout(sys.stderr):
                if request['op'] == 'reset':
                    controller.reset(state, refs, request['seed'])
                    response = {'reset': True}
                elif request['op'] == 'act':
                    response = {'command': asdict(controller.act(state, refs))}
                else:
                    raise ValueError('unknown worker operation')
            print(json.dumps(response, allow_nan=False), flush=True)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({'error': f'{type(error).__name__}: {error}'}), flush=True)
        raise SystemExit(1)
    finally:
        if controller is not None:
            with contextlib.redirect_stdout(sys.stderr):
                controller.close()


if __name__ == '__main__':
    main()
