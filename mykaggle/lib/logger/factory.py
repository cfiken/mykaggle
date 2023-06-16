from enum import Enum
from mykaggle.lib.logger.logger import Logger


class LoggerType(Enum):
    TENSORBOARD = 'tensorboard'
    MLFLOW = 'mlflow'
    STD = 'std'  # for debug


class LoggerFactory:

    @classmethod
    def create(cls, logger_type: LoggerType, *args, **kwargs) -> Logger:
        if logger_type == LoggerType.MLFLOW:
            from mykaggle.lib.logger.ml_logger import MLLogger
            return MLLogger(*args, **kwargs)
        elif logger_type == LoggerType.STD:
            from mykaggle.lib.logger.std_logger import StdLogger
            return StdLogger(*args, **kwargs)
        else:
            raise ValueError(f'{logger_type} is not defined.')
