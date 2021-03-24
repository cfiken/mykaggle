from typing import Any, Dict, Optional
from pathlib import Path
import yaml

from mykaggle.util.ml_logger import MLLogger


def save_settings(settings: Dict[str, Any], save_dir: Path, logger: Optional[MLLogger] = None) -> None:
    filepath = save_dir / 'settings.yml'

    def serialize(obj: Dict[str, Any]) -> Dict[str, Any]:
        new_obj: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, Dict):
                new_obj[k] = serialize(v)
            elif isinstance(v, (str, int, float, bool, list, tuple)):
                new_obj[k] = v
            else:
                new_obj[k] = str(v)
        return new_obj

    with open(filepath, 'w') as f:
        yaml.dump(serialize(settings), f)
    if logger is not None:
        logger.log_artifact(str(filepath))
