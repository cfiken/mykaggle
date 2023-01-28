from typing import Any, Dict, List, Callable
from pathlib import Path
import pickle
import gc
import numpy as np
import pandas as pd
import torch
from torch import nn

from mykaggle.util.logger import get_logger
from mykaggle.trainer.base import Mode
from mykaggle.trainer.get_trainer import get_trainer
from mykaggle.trainer.inference import load_oof
from mykaggle.trainer.loss.torch import get_loss_fn
from mykaggle.trainer.optimizer.torch import get_optimizer, get_scheduler
from mykaggle.dataloader.dataset import get_dataset
from mykaggle.dataloader.dataloader import get_dataloader

LOGGER = get_logger(__name__)


def get_stacking_model(settings: Dict[str, Any]):
    pass


def stacking_training(
    settings: Dict[str, Any],
    models: List[str],
    df: pd.DataFrame,
    folds: List,
    ckptdir: Path,
    metric: Callable,
    ml_logger
) -> np.ndarray:
    pred_dict = {}
    for model_name in models:
        oof_preds = load_oof(model_name, ckptdir.parent)
        pred_dict[model_name] = oof_preds
    df_st = pd.DataFrame(pred_dict)
    df_st['target'] = df['target']

    oof_preds = np.zeros((df_st.shape[0]))
    for fold, (train_idx, valid_idx) in enumerate(folds):
        gc.collect()
        torch.cuda.empty_cache()
        LOGGER.info(f'fold {fold + 1}/{len(folds)} start.')
        df_train = df_st.iloc[train_idx]
        df_valid = df_st.iloc[valid_idx]

        ds_train = get_dataset(settings['training'], Mode.TRAIN, df_train)
        ds_valid = get_dataset(settings['training'], Mode.VALID, df_valid)
        train_dataloader = get_dataloader(settings['training'], ds_train, Mode.TRAIN, fold)
        valid_dataloader = get_dataloader(settings['training'], ds_valid, Mode.VALID, fold)
        model = get_stacking_model(settings).cuda()
        loss_fn = get_loss_fn('mse', 1, loss_reduction='mean')
        optimizer = get_optimizer('Adam', settings['training']['learning_rate'], 0.0, model.parameters())
        scheduler = get_scheduler(settings['training'], optimizer)
        LOGGER.info('start training')
        trainer = get_trainer(settings['training']['trainer'], settings, ckptdir, ml_logger, fold)
        trainer.train(
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        model.load_state_dict(torch.load(ckptdir / f'model_{fold}.pt'))
        val_preds, _ = trainer.validation(valid_dataloader, model)
        score = metric(df_valid['target'].values, val_preds)
        LOGGER.info(f'metric_{fold}: {score}')
        ml_logger.log_metric(f'metric_{fold}', score)
        if valid_idx is not None:
            oof_preds[valid_idx] = val_preds
        else:
            oof_preds = val_preds
        del trainer, model, optimizer, scheduler, loss_fn

    score = metric(df_st['target'].values, oof_preds)
    ml_logger.log_metric('metric', score)
    pickle.dump(oof_preds, open(ckptdir / 'oof_preds.pkl', 'wb'))
    LOGGER.info(f'training finished. metric: {score:.3f}')
    return oof_preds


def stacking_inference(
    settings: Dict[str, Any],
    models: List[str],
    df: pd.DataFrame,
    ckptdir: Path,
    is_kaggle: bool
) -> np.ndarray:
    whole_preds = np.zeros((len(df)))
    return whole_preds
