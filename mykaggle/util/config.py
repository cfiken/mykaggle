import yaml
import json
from pathlib import Path
from typing import Any, Dict

BASEDIR = Path('./mykaggle/config/')


def load_config(path: Path, is_full_path: bool = False) -> Dict[str, Any]:
    '''Load config file from some kinds of format: yaml, json.
    distinguish format by filename extension, return Dict
    '''
    ext = path.suffix
    if not is_full_path:
        path = BASEDIR / path
    if ext == '.yml':
        return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    elif ext == '.json':
        return json.load(open(path, 'r'))
    else:
        raise ValueError('config file type is allowed only in ".yml" and ".json"')
