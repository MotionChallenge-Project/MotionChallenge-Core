"""Load an adapter factory from its locked repository environment."""
import importlib
import json
from pathlib import Path
import sys


def load_controller(config_path, entrypoint='entrypoint'):
    cfg = json.loads(Path(config_path).read_text())
    root = Path(cfg.get('adapter_root', Path(config_path).resolve().parent)).resolve()
    sys.path.insert(0, str(root))
    for path in cfg.get('python_paths', []):
        sys.path.insert(0, str((root/path).resolve()))
    module, factory = cfg[entrypoint].split(':')
    return getattr(importlib.import_module(module), factory)(root, cfg)
