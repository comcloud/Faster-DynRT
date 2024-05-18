import math

import torch
import torch.nn.functional as F
from torch import nn


class MHAtt(nn.Module):
    def __init__(self, q_hidden_dim, k_v_hidden_dim, head=1):
        super(MHAtt, self).__init__()
        self.head = head
        self.q_hidden_dim = q_hidden_dim
        self.k_v_hidden_dim = k_v_hidden_dim

        self.linear_v = nn.Linear(k_v_hidden_dim, k_v_hidden_dim)
        self.linear_k = nn.Linear(k_v_hidden_dim, k_v_hidden_dim)
        self.linear_q = nn.Linear(q_hidden_dim, q_hidden_dim)
        self.linear_merge = nn.Linear(q_hidden_dim, q_hidden_dim)

        self.dropout = nn.Dropout(0.5)

    def forward(self, q, k, v, mask=None):
        n_batches = q.size(0)

        v = self.linear_v(v).view(
            n_batches,
            -1,
            self.head,
            int(self.k_v_hidden_dim / self.head)
        ).transpose(1, 2)

        k = self.linear_k(k).view(
            n_batches,
            -1,
            self.head,
            int(self.k_v_hidden_dim / self.head)
        ).transpose(1, 2)

        q = self.linear_q(q).view(
            n_batches,
            -1,
            self.head,
            int(self.q_hidden_dim / self.head)
        ).transpose(1, 2)

        atted = self.att(v, k, q, mask)
        atted = atted.transpose(1, 2).contiguous().view(
            n_batches,
            -1,
            self.q_hidden_dim
        )

        atted = self.linear_merge(atted)

        return atted

    def att(self, value, key, query, mask):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        att_map = F.softmax(scores, dim=-1)
        att_map = self.dropout(att_map)

        return torch.matmul(att_map, value)
