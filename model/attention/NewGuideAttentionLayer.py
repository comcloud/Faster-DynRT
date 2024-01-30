import torch
from torch import nn

import math

'''
分别自注意力
'''


class BothGuideAttentionLayer(nn.Module):

    def __init__(self, batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim):
        super(BothGuideAttentionLayer, self).__init__()
        self.batch_size = batch_size

        # sparse attention
        self.text_sparse_attention = LocalAttention(text_hidden_dim)
        self.image_sparse_attention = LocalAttention(image_hidden_dim)

        # MLP
        self.text_out = nn.Sequential(nn.Linear(text_hidden_dim, text_hidden_dim * 4), nn.ReLU(),
                                      nn.Linear(text_hidden_dim * 4, text_hidden_dim))
        self.image_out = nn.Sequential(nn.Linear(image_hidden_dim, image_hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(image_hidden_dim * 4, image_hidden_dim))

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # norm
        self.text_norm = nn.LayerNorm(
            normalized_shape=[batch_size, text_seq_len, text_hidden_dim],
            eps=1e-6,
            device=self.device)
        self.image_norm = nn.LayerNorm(
            normalized_shape=[batch_size, image_block_num, image_hidden_dim],
            eps=1e-6,
            device=self.device)

    def forward(self, text_feature, image_feature):
        # 先对图片进行处理
        # 1. sparse attention
        image_out = self.image_sparse_attention(image_feature)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)
        # 2. MLP
        image_out = self.image_out(image_out)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)

        # 再对文本处理
        # 1. sparse attention
        text_out = self.text_sparse_attention(text_feature)
        # norm
        text_out = self.norm(text_feature, text_out, self.text_norm)
        # 2. MLP out
        text_out = self.text_out(text_out)
        # norm
        text_out = self.norm(text_feature, text_out, self.text_norm)

        return text_out, image_out

    def norm(self, feature, out, norm_manner):
        '''
        归一化
        :param feature: 初始特征
        :param out: 一次网络层处理结果
        :param norm_manner: norm方式
        '''
        # 残差后归一化
        if self.batch_size == feature.size(0):
            return norm_manner(out + feature)

        # 处理后续数据集不足batch size的情况
        norm_supple = nn.LayerNorm(
            normalized_shape=[feature.size(0), feature.size(1), feature.size(2)],
            eps=1e-6,
            device=self.device)
        return norm_supple(out + feature)


class TextGuideAttentionLayer(nn.Module):

    def __init__(self, batch_size, text_seq_len, text_hidden_dim):
        super(TextGuideAttentionLayer, self).__init__()
        self.batch_size = batch_size

        # sparse attention
        self.text_sparse_attention = LocalAttention(text_hidden_dim)

        # MLP
        self.text_out = nn.Sequential(nn.Linear(text_hidden_dim, text_hidden_dim * 4), nn.ReLU(),
                                      nn.Linear(text_hidden_dim * 4, text_hidden_dim))

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # norm
        self.text_norm = nn.LayerNorm(
            normalized_shape=[batch_size, text_seq_len, text_hidden_dim],
            eps=1e-6,
            device=self.device)

    def forward(self, text_feature, image_feature):
        # 1. sparse attention
        text_out = self.text_sparse_attention(text_feature)
        # norm
        text_out = self.norm(text_feature, text_out, self.text_norm)
        # 2. MLP out
        text_out = self.text_out(text_out)
        # norm
        text_out = self.norm(text_feature, text_out, self.text_norm)

        return text_out, image_feature

    def norm(self, feature, out, norm_manner):
        '''
        归一化
        :param feature: 初始特征
        :param out: 一次网络层处理结果
        :param norm_manner: norm方式
        '''
        # 残差后归一化
        if self.batch_size == feature.size(0):
            return norm_manner(out + feature)

        # 处理后续数据集不足batch size的情况
        norm_supple = nn.LayerNorm(
            normalized_shape=[feature.size(0), feature.size(1), feature.size(2)],
            eps=1e-6,
            device=self.device)
        return norm_supple(out + feature)


class ImageGuideAttentionLayer(nn.Module):

    def __init__(self, batch_size, image_block_num, image_hidden_dim):
        super(ImageGuideAttentionLayer, self).__init__()
        self.batch_size = batch_size

        # sparse attention
        self.image_sparse_attention = LocalAttention(image_hidden_dim)

        # MLP
        self.image_out = nn.Sequential(nn.Linear(image_hidden_dim, image_hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(image_hidden_dim * 4, image_hidden_dim))

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # norm
        self.image_norm = nn.LayerNorm(
            normalized_shape=[batch_size, image_block_num, image_hidden_dim],
            eps=1e-6,
            device=self.device)

    def forward(self, text_feature, image_feature):
        # 1. sparse attention
        image_out = self.image_sparse_attention(image_feature)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)
        # 2. MLP
        image_out = self.image_out(image_out)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)

        return text_feature, image_out

    def norm(self, feature, out, norm_manner):
        '''
        归一化
        :param feature: 初始特征
        :param out: 一次网络层处理结果
        :param norm_manner: norm方式
        '''
        # 残差后归一化
        if self.batch_size == feature.size(0):
            return norm_manner(out + feature)

        # 处理后续数据集不足batch size的情况
        norm_supple = nn.LayerNorm(
            normalized_shape=[feature.size(0), feature.size(1), feature.size(2)],
            eps=1e-6,
            device=self.device)
        return norm_supple(out + feature)


class LocalAttention(nn.Module):
    def __init__(self, hidden_dim, window_size=2):
        super(LocalAttention, self).__init__()
        self.window_size = window_size
        # self.n_heads = n_heads

        # 图片作为Q
        # 定义 Q, K, V 的线性映射
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    def forward(self, feature):
        d_k = feature.size(-1)
        # 线性映射得到 Q, K, V
        q = self.q_linear(feature)
        k = self.k_linear(feature)
        v = self.v_linear(feature)

        two_pos = feature.size(1)
        # 定义局部注意力分布
        attention_mask = torch.zeros(two_pos, two_pos)
        attention_mask = attention_mask.to(self.device)
        for i in range(two_pos):
            start = max(0, i - self.window_size)
            end = min(two_pos, i + self.window_size + 1)
            attention_mask[i, start:end] = 1

        # 计算注意力权重
        attention_weights = torch.softmax(attention_mask, dim=-1)

        # 加权求和
        # 这里单独写出来batch size是因为存在数据集最后一部分没有够一个batch size
        batch_size = feature.size(0)
        attention_weights = attention_weights.unsqueeze(0).repeat(batch_size, 1, 1)
        output = torch.bmm(attention_weights, k)  # (batch_size, seq_len, text_hidden_dim)
        output = torch.bmm(q, output.transpose(2, 1)) / math.sqrt(d_k)  # (batch_size, 1, seq_len)
        output = torch.bmm(torch.softmax(output, dim=-1), v)  # (batch_size, 1, text_hidden_dim)

        return output
