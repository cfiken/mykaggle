from typing import Any, Dict
import torch
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel

from mykaggle.model.common import (
    AttentionHead, LayerAttentionPooling
)


class LMModelWithCustomHead(nn.Module):
    def __init__(
        self,
        settings: Dict[str, Any],
        model: PreTrainedModel
    ) -> None:
        super().__init__()
        self.settings = settings
        self.model = model
        self.head_types = settings['model']['custom_head_types']
        self.num_layers_for_output = self.settings['model']['num_use_layers_for_output']
        self.num_reinit_layers = settings['model'].get('num_reinit_layers', 0)
        output_layers = {}

        if 'attn' in self.head_types:
            self.hidden_dim = self.settings['model']['head_hidden_dim']
            self.intermediate_dim = self.settings['model'].get('head_intermediate_dim', self.hidden_dim)
            self.attn_head = AttentionHead(self.hidden_dim, self.intermediate_dim)
        if 'conv' in self.head_types:
            hidden_dim = self.settings['model'].get('conv_head_hidden_dim', 256)
            kernel_size = self.settings['model'].get('conv_head_kernel_size', 2)
            self.conv1 = nn.Conv1d(self.model.config.hidden_size, hidden_dim, kernel_size=kernel_size, padding=1)
            self.conv2 = nn.Conv1d(hidden_dim, 1, kernel_size=kernel_size, padding=1)
        if 'layers_sum' in self.head_types:
            self.layer_weight = nn.Parameter(torch.tensor([1] * self.num_layers_for_output, dtype=torch.float))
        if 'layers_attn' in self.head_types:
            self.layer_attn = LayerAttentionPooling(self.settings['model']['head_hidden_dim'])

        for head in self.head_types:
            if 'concat' in head:
                output_layers[head] = nn.Linear(self.model.config.hidden_size * self.num_layers_for_output, 1)
            elif head == 'conv':
                continue
            else:
                output_layers[head] = nn.Linear(self.model.config.hidden_size, 1)
        self.output_layers = nn.ModuleDict(output_layers)

        self.dropout = nn.Dropout(settings['model']['dropout_rate'])
        self.ensemble_type = settings['model']['custom_head_ensemble']
        if self.ensemble_type == 'weight':
            self.ensemble_weight = nn.Linear(len(self.head_types), 1, bias=False)
        self.output_head_features = settings['model'].get('output_head_features', False)

        self.initialize()

    def forward(self, inputs):
        outputs = self.model(**inputs, output_hidden_states=True)
        head_features = []
        features = []
        if 'cls' in self.head_types:
            cls_state = outputs.last_hidden_state[:, 0, :]
            feature = self.output_layers['cls'](self.dropout(cls_state))
            head_features.append(cls_state)
            features.append(feature)
        if 'avg' in self.head_types:
            # input_mask = inputs['attention_mask'].unsqueeze(-1).float()
            # sum_embeddings = torch.sum(outputs.last_hidden_state * input_mask, 1)
            # avg_pool = sum_embeddings / torch.sum(input_mask, 1)
            avg_pool = torch.mean(outputs.last_hidden_state, 1)
            feature = self.output_layers['avg'](self.dropout(avg_pool))
            head_features.append(avg_pool)
            features.append(feature)
        if 'max' in self.head_types:
            # input_mask = inputs['attention_mask'].unsqueeze(-1).float()
            # max_pool = torch.max(outputs.last_hidden_state * input_mask, 1)[0]
            max_pool = torch.max(outputs.last_hidden_state, 1)[0]
            feature = self.output_layers['max'](self.dropout(max_pool))
            head_features.append(max_pool)
            features.append(feature)
        if 'attn' in self.head_types:
            attn_state = self.attn_head(outputs.last_hidden_state)
            feature = self.output_layers['attn'](self.dropout(attn_state))
            head_features.append(attn_state)
            features.append(feature)
        if 'conv' in self.head_types:
            conv_state = self.conv1(outputs.last_hidden_state.permute(0, 2, 1))
            conv_state = F.relu(self.conv2(conv_state))
            feature, _ = torch.max(conv_state, -1)
            head_features.append(conv_state)
            features.append(feature)
        if 'layers_concat' in self.head_types:
            hidden_states = outputs.hidden_states[-self.num_layers_for_output:]
            cat_feature = torch.cat([state[:, 0, :] for state in hidden_states], -1)
            feature = self.output_layers['layers_concat'](self.dropout(cat_feature))
            head_features.append(cat_feature)
            features.append(feature)
        if 'layers_avg' in self.head_types:
            hidden_states = torch.stack(outputs.hidden_states[-self.num_layers_for_output:], -1)[:, 0, :, :]
            avg_feature = torch.mean(hidden_states, -1)
            feature = self.output_layers['layers_avg'](self.dropout(avg_feature))
            head_features.append(avg_feature)
            features.append(feature)
        if 'layers_sum' in self.head_types:
            hidden_states = torch.stack(outputs.hidden_states[-self.num_layers_for_output:], -1)[:, 0, :, :]
            weight = self.layer_weight[None, None, :] / self.layer_weight.sum()
            weighted_sum_feature = torch.sum(hidden_states * weight, -1)
            feature = self.output_layers['layers_sum'](self.dropout(weighted_sum_feature))
            head_features.append(weighted_sum_feature)
            features.append(feature)
        if 'layers_attn' in self.head_types:
            hidden_states = torch.stack(outputs.hidden_states[-self.num_layers_for_output:], -1)[:, 0, :, :]
            attn_state = self.layer_attn(hidden_states)
            feature = self.output_layers['layers_attn'](self.dropout(attn_state))
            head_features.append(attn_state)
            features.append(feature)

        outputs = torch.cat(features, -1)
        if len(self.head_types) > 1:
            if self.ensemble_type == 'avg':
                outputs = torch.mean(outputs, -1)
            elif self.ensemble_type == 'weight':
                if self.settings['training']['trainer'] == 'multi':
                    outputs = outputs.detach()
                weight = self.ensemble_weight.weight / torch.sum(self.ensemble_weight.weight)
                outputs = torch.sum(weight * outputs, -1)
        outputs = outputs.reshape(inputs['input_ids'].shape[0])
        if self.output_head_features:
            features = [f.reshape(inputs['input_ids'].shape[0]) for f in features]
            return outputs, features, head_features
        return outputs

    def initialize(self):
        if self.ensemble_type == 'weight':
            torch.nn.init.constant_(self.ensemble_weight.weight, 1.0)
        self.output_layers.apply(self._init_weight)
        for i in range(self.num_reinit_layers):
            self.model.encoder.layer[-(1 + i)].apply(self._init_weight)

    def _init_weight(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
