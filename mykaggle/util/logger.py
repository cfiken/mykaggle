import sys
import logging


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    '''
    ライブラリ側で使用する logger を取得します。
    :param name: logger の名前空間
    '''
    logger = logging.getLogger(name)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
