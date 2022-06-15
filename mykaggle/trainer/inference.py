from typing import Any, Dict, List, Optional, Callable
from pathlib import Path
import gc
import pandas as pd
import numpy as np
import yaml
import pickle
from scipy.optimize import minimize
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from mykaggle.util.logger import get_logger
from mykaggle.model.model import get_model
from mykaggle.trainer.base import Mode
from mykaggle.dataloader.dataloader import get_dataloader
from mykaggle.dataloader.dataset import get_dataset

LOGGER = get_logger(__name__)


def predict(
    model: nn.Module,
    df: pd.DataFrame,
    dataloader: DataLoader,
    batch_size: int,
    num_classes: int,
    use_amp: bool = True
) -> np.ndarray:
    if num_classes > 1:
        preds = np.zeros((len(df), num_classes), dtype=np.float32)
    else:
        preds = np.zeros((len(df)), dtype=np.float32)
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    for i, batch in enumerate(dataloader):
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
        else:
            inputs = batch
        for key in inputs.keys():
            inputs[key] = inputs[key].to(device).long()
        with autocast(enabled=use_amp):
            with torch.no_grad():
                outputs = model(inputs)
                if isinstance(outputs, (list, tuple)):
                    outputs = outputs[0]
        preds[i * batch_size:(i + 1) * batch_size] = outputs.detach().cpu().numpy()
    return preds


def validation(
    settings: Dict[str, Any],
    model: nn.Module,
    dataloader: DataLoader,
    df_valid: pd.DataFrame,
) -> np.ndarray:
    batch_size = settings['training']['test_batch_size']
    use_amp = settings['training']['use_amp']
    num_classes = settings['training']['num_classes']
    preds = predict(model, df_valid, dataloader, batch_size, num_classes, use_amp)
    return preds


def test(
    settings: Dict[str, Any],
    model: nn.Module,
    dataloader: DataLoader,
    df_test: pd.DataFrame,
) -> np.ndarray:
    batch_size = settings['training']['test_batch_size']
    use_amp = settings['training']['use_amp']
    num_classes = settings['training']['num_classes']
    preds = predict(model, df_test, dataloader, batch_size, num_classes, use_amp)
    return preds


def ensemble_oof(models: List[str], df: pd.DataFrame, num_classes: int, ckptdir: Path) -> np.ndarray:
    if num_classes > 1:
        oof_preds = np.zeros((len(models), len(df), num_classes))
    else:
        oof_preds = np.zeros((len(models), len(df)))
    for i, model_name in enumerate(models):
        _oof_preds = load_oof(model_name, ckptdir)
        oof_preds[i] = _oof_preds
    LOGGER.info(f'ensembled oof with {",".join(models)} finished.')
    return oof_preds


def ensemble_inference(
    models: List[str],
    df: pd.DataFrame,
    num_classes: int,
    ckptdir: Path,
    is_kaggle: bool,
    ensemble_type: str = 'avg',
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    device = torch.device('cuda')

    whole_preds = np.zeros((len(models), len(df)))
    for i, model_name in enumerate(models):
        gc.collect()
        torch.cuda.empty_cache()
        LOGGER.info(f'inference by {model_name} start.')
        model_ckptdir = ckptdir / model_name
        model_settings = yaml.safe_load(open(model_ckptdir / 'settings.yml', 'r'))
        model_settings['ckptdir'] = model_ckptdir
        model_settings['training']['ckptdir'] = model_ckptdir
        model_settings['model']['ckpt_from'] = None
        for fold in range(model_settings['training']['num_folds']):
            gc.collect()
            torch.cuda.empty_cache()
            ds = get_dataset(model_settings['training'], Mode.TEST, df)
            dataloader = get_dataloader(model_settings['training'], ds, Mode.TEST, fold)
            model = get_model(model_settings, fold=fold, is_kaggle=is_kaggle, pretrained=False)
            model.load_state_dict(torch.load(model_ckptdir / f'model_{fold}.pt'))
            model.to(device)
            preds = test(model_settings, model, dataloader, df)
            whole_preds[i] += preds / model_settings['training']['num_folds']
            del model
    if ensemble_type == 'avg':
        whole_preds = np.mean(whole_preds, axis=0)
    else:
        if weights is not None:
            whole_preds = np.sum(whole_preds * weights[:, np.newaxis], axis=0)
        else:
            whole_preds = np.mean(whole_preds, axis=0)

    return whole_preds


def ensemble_inference_fold(
    models: List[str],
    df: pd.DataFrame,
    ckptdir: Path,
    fold: int,
    is_kaggle: bool,
    ensemble_type: str = 'avg',
    weights: Optional[np.ndarray] = None
) -> np.ndarray:
    device = torch.device('cuda')

    whole_preds = np.zeros((len(models), len(df)))
    for i, model_name in enumerate(models):
        gc.collect()
        torch.cuda.empty_cache()
        LOGGER.info(f'inference by {model_name} start.')
        model_ckptdir = ckptdir / model_name
        model_settings = yaml.safe_load(open(model_ckptdir / 'settings.yml', 'r'))
        model_settings['ckptdir'] = model_ckptdir
        model_settings['training']['ckptdir'] = model_ckptdir
        model_settings['model']['ckpt_from'] = None

        ds = get_dataset(model_settings['training'], Mode.TEST, df)
        dataloader = get_dataloader(model_settings['training'], ds, Mode.TEST, fold)
        model = get_model(model_settings, fold=fold, is_kaggle=is_kaggle, pretrained=False)
        model.load_state_dict(torch.load(model_ckptdir / f'model_{fold}.pt'))
        model.to(device)
        preds = test(model_settings, model, dataloader, df)
        whole_preds[i] = preds
        del model
    if ensemble_type == 'avg':
        whole_preds = np.mean(whole_preds, axis=0)
    else:
        if weights is not None:
            whole_preds = np.sum(whole_preds * weights[:, np.newaxis], axis=0)
        else:
            whole_preds = np.mean(whole_preds, axis=0)

    return whole_preds


def train_ensemble_weights(
    ensemble_type: str,
    targets: np.ndarray,
    preds: np.ndarray,
    metric: Callable,
    lbs: Optional[np.ndarray] = None
):
    num_models = preds.shape[0]
    if ensemble_type == 'avg':
        return np.mean(preds, axis=0), None
    else:
        def loss_weight_ensemble(weights, targets, preds, lbs):
            ensembled_preds = 0
            for w, p in zip(weights, preds):
                ensembled_preds += w * p
            if lbs is not None:
                lb_weight = np.mean(np.abs(weights) * lbs)
                return metric(targets, ensembled_preds) + lb_weight
            return metric(targets, ensembled_preds)

        cons = ({'type': 'eq', 'fun': lambda w: 1 - np.sum(np.abs(w))})
        bounds = [(0, 1)] * num_models
        weights_start = np.ones(num_models) / num_models

        result = minimize(
            loss_weight_ensemble,
            weights_start,
            args=(targets, preds, lbs),
            constraints=cons,
            bounds=bounds,
            method=ensemble_type
        )

        preds = np.sum(preds * result['x'][:, np.newaxis], axis=0)
        return preds, result['x']


def load_oof(model_name: str, ckptdir: Path) -> np.ndarray:
    model_ckptdir = ckptdir / model_name
    return pickle.load(open(model_ckptdir / 'oof_preds.pkl', 'rb'))
