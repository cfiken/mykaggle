from typing import Optional
import math
import torch
from torch import nn
from torch.nn import functional as F

from mykaggle.model.activation import get_activation


class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.k_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        self.k_dense_layer = nn.Linear(hidden_dim, hidden_dim)
        self.v_dense_layer = nn.Linear(hidden_dim, hidden_dim)
        self.q_dense_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query, key, attn_mask) -> torch.Tensor:
        q = self.split_head(self.q_dense_layer(query))
        k = self.split_head(self.k_dense_layer(key))
        v = self.split_head(self.v_dense_layer(key))

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights / math.sqrt(self.k_dim)

        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, -6e4)
        attn_weights = F.softmax(attn_weights, dim=-1)

        if self.dropout > 0:
            attn_weights = self.dropout_layer(attn_weights)

        outputs = torch.matmul(attn_weights, v)
        outputs = self.combine_head(outputs)
        outputs = self.output_layer(outputs)
        return outputs

    def split_head(self, value):
        batch_size = value.shape[0]
        value = value.reshape(batch_size, -1, self.num_heads, self.k_dim)
        return value.transpose(1, 2)

    def combine_head(self, value):
        batch_size = value.shape[0]
        value = value.transpose(1, 2)
        return value.reshape(batch_size, -1, self.hidden_dim)


class FeedForward(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.activation(self.linear1(inputs))
        outputs = self.linear2(self.dropout(outputs))
        return outputs


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
        self.ffn = FeedForward(hidden_dim, ffn_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, inputs, attn_mask):
        outputs = self.layer_norm1(inputs)
        outputs = self.attn(outputs, outputs, attn_mask)
        attn_outputs = inputs + self.dropout1(outputs)

        outputs = self.layer_norm2(attn_outputs)
        outputs = self.ffn(outputs)
        outputs = attn_outputs + self.dropout2(outputs)
        return outputs


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_layers: int
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, ffn_dim, num_heads)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, inputs, attn_mask):
        outputs = inputs
        for i in range(self.num_layers):
            outputs = self.layers[i](outputs, attn_mask)
        outputs = self.norm(outputs)
        return outputs


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)

        self.attn1 = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
        self.attn2 = MultiHeadAttention(hidden_dim, num_heads, dropout=0.1)
        self.ffn = FeedForward(hidden_dim, ffn_dim)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query, mem, attn_mask, mem_mask):
        outputs = self.norm1(query)
        outputs = self.attn1(outputs, outputs, attn_mask)  # self-attention
        first_outputs = query + self.dropout1(outputs)

        if mem is not None:
            outputs = self.norm2(first_outputs)
            outputs = self.attn2(outputs, mem, mem_mask)  # encoder input
            second_outputs = first_outputs + self.dropout2(outputs)

        outputs = self.norm3(second_outputs)
        outputs = self.ffn(outputs)
        outputs = second_outputs + self.dropout3(outputs)
        return outputs

    def forward_last(self, query_last, query_cache, mem, mem_mask: Optional = None):
        query_last_norm = self.norm1(query_last)
        outputs = torch.cat([query_cache, query_last_norm], 1)
        query_cache = outputs.clone()

        outputs = self.attn1(query_last_norm, outputs, None)
        first_outputs = query_last + outputs

        if mem is not None:
            outputs = self.norm2(first_outputs)
            outputs = self.attn2(outputs, mem, mem_mask)
            second_outputs = first_outputs + outputs

        outputs = self.norm3(second_outputs)
        outputs = self.ffn(outputs)
        last_outputs = second_outputs + outputs

        return last_outputs, query_cache


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_dim, ffn_dim, num_heads)
            for i in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, query, mem, attn_mask=None, mem_mask=None):
        for i in range(self.num_layers):
            query = self.layers[i](query, mem, attn_mask, mem_mask)

        outputs = self.norm(query)
        return outputs

    def forward_last(self, query_last, query_cache, mem, mem_mask=None):
        batch_size, t, dim = query_last.shape
        assert t == 1, 'process step by step in inference'

        for i in range(self.num_layers):
            query_last, query_cache[i] = self.layers[i].forward_last(
                query_last, query_cache[i], mem, mem_mask
            )

        query_last = self.norm(query_last)
        return query_last, query_cache
