import os
import random
import numpy as np
import tensorflow as tf
import torch


def initialize_gpu(gpu_ids: str, use_mixed_precision: bool = False, jit: bool = False) -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids
    devices = tf.config.experimental.list_physical_devices('GPU')
    for d in devices:
        tf.config.experimental.set_memory_growth(d, True)
    tf.config.optimizer.set_jit(jit)
    tf.config.optimizer.set_experimental_options({'auto_mixed_precision': use_mixed_precision})


def fix_seed(seed: int = 1019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
