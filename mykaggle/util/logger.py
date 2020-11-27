from logging import getLogger, Logger


def get_logger(name: str) -> Logger:
    '''
    ライブラリ側で使用する logger を取得します。
    :param name: logger の名前空間
    '''
    return getLogger(name)
