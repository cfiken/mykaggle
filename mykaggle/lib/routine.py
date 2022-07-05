'''
コンペでよく使うようなちょっとした処理の置き場所
'''

from typing import Optional, Any, Dict
from pathlib import Path
import os
import sys
import random
from time import time
from contextlib import contextmanager
from argparse import ArgumentParser, Namespace
import yaml
import json
import logging
import numpy as np
import torch
try:
    import tensorflow as tf
except ImportError:
    from unittest.mock import MagicMock
    tf = MagicMock()


def parse() -> Namespace:
    parser = ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--gpus', type=str, help='index of gpus. if multiple, use comma to list.'
    )
    args = parser.parse_args()
    return args


def load_config(path: Path, is_full_path: bool = False) -> Dict[str, Any]:
    '''Load config file from some kinds of format: yaml, json.
    distinguish format by filename extension, return Dict
    '''
    ext = path.suffix
    if not is_full_path:
        basedir = Path('./mykaggle/config/')
        path = basedir / path
    if ext == '.yml':
        return yaml.load(open(path, 'r'), Loader=yaml.FullLoader)
    elif ext == '.json':
        return json.load(open(path, 'r'))
    else:
        raise ValueError('config file type is allowed only in ".yml" and ".json"')


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


def fix_seed(seed: int = 1019) -> None:
    '''
    各種ライブラリで Seed を固定します。
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore


@contextmanager
def timer(
    logger: Optional[logging.Logger] = None,
    fmt: str = '{:.3f}[s]',
    prefix: Optional[str] = None,
    suffix: Optional[str] = None
):
    '''
    関数に付加して実行時間を計測し出力する context manager
    '''
    if prefix is not None:
        fmt = str(prefix) + fmt
    if suffix is not None:
        fmt = fmt + str(suffix)
    start = time()
    yield
    elapsed = time() - start
    out_str = fmt.format(elapsed)
    if logger:
        logger.info(out_str)
    else:
        print(out_str)


# If using TensorFlow

def initialize_gpu(gpu_ids: str, use_mixed_precision: bool = False, jit: bool = False) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    devices = tf.config.experimental.list_physical_devices('GPU')
    for d in devices:
        tf.config.experimental.set_memory_growth(d, True)
    tf.config.optimizer.set_jit(jit)
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': use_mixed_precision})
