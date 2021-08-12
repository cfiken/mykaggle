import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim

        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)

        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class LayerAttentionPooling(nn.Module):
    def __init__(self, hidden_size: int):
        super(LayerAttentionPooling, self).__init__()
        self.hidden_size = hidden_size

        q_t = np.random.normal(loc=0.0, scale=0.1, size=(1, self.hidden_size))
        self.q = nn.Parameter(torch.from_numpy(q_t)).float().cuda()
        w_ht = np.random.normal(loc=0.0, scale=0.1, size=(self.hidden_size, self.hidden_size))
        self.w_h = nn.Parameter(torch.from_numpy(w_ht)).float().cuda()

    def forward(self, hidden_states):
        hidden_states = hidden_states.permute(0, 2, 1)  # [batch_size, num_use_layers, hidden_size]
        out = self.attention(hidden_states)
        return out

    def attention(self, hidden_states):
        v = torch.matmul(self.q, hidden_states.transpose(-2, -1)).squeeze(1)
        v = F.softmax(v, -1)
        v_temp = torch.matmul(v.unsqueeze(1), hidden_states).transpose(-2, -1)
        v = torch.matmul(self.w_h.transpose(1, 0), v_temp).squeeze(2)
        return v
