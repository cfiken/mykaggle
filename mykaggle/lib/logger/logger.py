from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional, Generator
from pathlib import Path
import contextlib


class StatusType(Enum):
    RUNNING = 'RUNNING'
    SUCCESS = 'SUCCESS'
    FAILURE = 'FAILURE'


class Logger(ABC):
    def __init__(self, user: str, logdir: Path) -> None:
        self._user = user
        self._logdir = logdir

    @abstractmethod
    @contextlib.contextmanager
    def start(self, experiment_name: str, run_name: str) -> Generator:
        raise NotImplementedError()

    @abstractmethod
    def log_params(self, params: dict[str, Any]):
        raise NotImplementedError()

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def log_metric(self, name: str, metric: float, step: int = 0) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save_model(
        self,
        model,
        filename: str = 'model',
        artifact_path: Optional[str] = None,
        upload: bool = False
    ):
        raise NotImplementedError()

    @abstractmethod
    def upload_model(self, filename: str, artifact_path: Optional[str] = None) -> None:
        raise NotImplementedError()

    @abstractmethod
    def save_config(self, settings: dict[str, Any]) -> None:
        raise NotImplementedError()
