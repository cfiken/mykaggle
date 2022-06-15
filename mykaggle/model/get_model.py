from typing import Any, Dict, Optional
from pathlib import Path
import torch
from torch import nn
from transformers import (
    PreTrainedModel, AutoConfig, AutoModel,
    BertModel, RobertaModel, ElectraModel, DebertaModel,
    DebertaV2Model, XLMModel, AlbertModel,
)
from mykaggle.model.lm_model_with_custom_head import LMModelWithCustomHead
from mykaggle.util.logger import get_logger

LOGGER = get_logger(__name__)


def get_model(
    settings: Dict[str, Any],
    fold: Optional[int] = None,
    is_kaggle: bool = False,
    pretrained: bool = True,
    hg_model: Optional[PreTrainedModel] = None,
    *args, **kwargs
) -> nn.Module:
    mst = settings['model']
    fold = fold or 0
    if hg_model is None:
        hg_model = get_transformers_model(
            settings['model'], mst['model_name'], pretrained, ckptdir=settings['ckptdir']
        )
    if fold == 0:
        LOGGER.info(hg_model.config)  # type: ignore
    model_type = settings['model'].get('model_type', 'cls_base')
    model: nn.Module
    if model_type == 'base':
        model = LMModelWithCustomHead(settings, hg_model)
    else:
        model = LMModelWithCustomHead(settings, hg_model)

    if not is_kaggle and mst.get('ckpt_from') is not None and mst.get('ckpt_from') != 'None':
        ckpt_from = mst['ckpt_from']
        if '{fold}' in ckpt_from and fold is not None:
            ckpt_from = ckpt_from.replace('{fold}', str(fold))
        LOGGER.info(f'load model from {ckpt_from}')
        state_dict = torch.load(mst['ckpt_from'], map_location='cpu')
        status = model.load_state_dict(state_dict, strict=False)  # type: ignore
        LOGGER.info(status)
        del state_dict

    # re init if needed
    model.initialize()  # type: ignore
    return model


def get_transformers_model(
    settings: Dict[str, Any],
    model_name: str,
    pretrained: bool = True,
    ckptdir: Optional[Path] = None,
) -> PreTrainedModel:
    model_path = model_name if pretrained else str(ckptdir)
    config = AutoConfig.from_pretrained(model_path)
    config.attention_probs_dropout_prob = settings.get('encoder_attn_dropout_rate', 0.1)
    config.hidden_dropout_prob = settings.get('encoder_ffn_dropout_rate', 0.1)
    config.layer_norm_eps = settings.get('layer_norm_eps', 1e-5)

    if pretrained:
        model = AutoModel.from_pretrained(model_name, config=config)
        return model

    # if you want not parameters but only model structure, each model class is needed.
    if 'xlm' in model_name:
        model = XLMModel(config=config)
    elif 'albert' in model_name:
        model = AlbertModel(config=config)
    elif 'roberta' in model_name:
        model = RobertaModel(config=config)
    elif 'deberta-v2' in model_name:
        model = DebertaV2Model(config=config)
    elif 'deberta' in model_name:
        model = DebertaModel(config=config)
    elif 'bert' in model_name:
        model = BertModel(config=config)
    elif 'electra' in model_name:
        model = ElectraModel(config=config)
    else:
        model = BertModel(config=config)
    return model
