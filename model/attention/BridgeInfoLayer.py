'''
桥梁信息层
主要用来构建文本和图像属性以及图像与图像属性的之间构造关系
'''
import math

import torch.nn
import torch.nn.functional as F
from torch import nn


class BridgeInfoLayer(torch.nn.Module):
    def __init__(self, seq_len, block_num):
        super(BridgeInfoLayer, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.text_biaffine = Biaffine(seq_len, seq_len, seq_len).to(self.device)
        self.image_biaffine = Biaffine(block_num, seq_len, block_num).to(self.device)

    def forward(self, text_feature, att_feature, image_feature):
        # 文属
        text_feature = self.text_biaffine(text_feature, att_feature)
        # 图属
        image_feature = self.image_biaffine(image_feature, att_feature)

        # 跨模态
        # text_feature = self.cross_attention(text_feature, image_feature, image_feature)
        # image_feature = self.cross_attention(image_feature, text_feature, text_feature)
        return text_feature, image_feature


class Biaffine(nn.Module):

    def __init__(self, in_size_f, in_size_a, out_size):
        # in_size_f : 文本或图像特征数
        # in_size_a : 图像属性特征数
        # out_size : 目标特征数
        super(Biaffine, self).__init__()
        self.in_size_f = in_size_f
        self.in_size_a = in_size_a
        self.out_size = out_size
        self.in_f_out_ln = nn.Linear(in_size_f, out_size)
        self.in_a_out_ln = nn.Linear(in_size_a, out_size)

    def forward(self, x, y):
        """
        batch_size,seq_len,hidden=x.shape
        bilinar_mapping=torch.matmul(x,self.U)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len*self.out_size,hidden))
        y=torch.transpose(y,dim0=1,dim1=2)
        bilinar_mapping=torch.matmul(bilinar_mapping,y)
        bilinar_mapping=bilinar_mapping.view(size=(batch_size,seq_len,self.out_size,seq_len))
        bilinar_mapping=torch.transpose(bilinar_mapping,dim0=2,dim1=3)
        """
        # bilinar_mapping = torch.einsum('bxi,ioj,byj->bxyo', x, self.U, y)

        x, y = self.in_f_out_ln(x.transpose(-2, -1)).transpose(-2, -1), \
               self.in_a_out_ln(y.transpose(-2, -1)).transpose(-2, -1)
        z = x + y

        bilinar_mapping = torch.bmm(x, y.transpose(-2, -1))
        bilinar_mapping = torch.bmm(bilinar_mapping, z)
        return bilinar_mapping


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


if __name__ == '__main__':
    t_f = torch.randn(2, 100, 768)
    a_f = torch.randn(2, 100, 768)
    i_f = torch.randn(2, 49, 768)

    # biaffine = Biaffine(49, 100, 49)
    # res = biaffine(i_f, a_f)
    # print(res.size())

    model = BridgeInfoLayer(100, 49)
    t_f, i_f = model(t_f, a_f, i_f)
    # t_f = torch.mean(t_f, dim=1)
    # i_f = torch.mean(i_f, dim=1)
    print(t_f.size())
    print(i_f.size())
