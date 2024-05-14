from enum import Enum

import torch
from torch import nn

from model.attention.ImageSparseAttention import ImageSparseAttention
from model.attention.TextSparseAttention import TextSparseAttention

# from GatedLayer import GatedLayer

'''
引导注意力层GuideAttentionLayer
中间使用策略层去处理不同的情况Strategy
真正执行者DoGuideAttentionLayer

'''


class GuideAttention(nn.Module):

    def __init__(self, batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim, use_source):
        super(GuideAttention, self).__init__()
        self.use_source = use_source
        self.strategy = Strategy(batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim)

    def forward(self, text_feature, image_feature):
        text_out, image_out = self.strategy(text_feature, image_feature, self.use_source)
        return text_out, image_out


class Strategy(nn.Module):

    def __init__(self, batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim, *args, **kwargs):
        # 初始化时注册所有处理策略
        # 0 : 使用原生的文本或者图片特征
        # 1 : 使用文本引导图片的结果作为图片和文本中的文本特征
        # 2 : 使用图片引导文本的结果作为图片和文本中的图片特征
        # 3 : 文本和图片都进行优化
        super().__init__(*args, **kwargs)
        self.strategies = {
            0: self.process_strategy_0,
            1: self.process_strategy_1,
            2: self.process_strategy_2,
            3: self.process_strategy_3,
        }

        self.do_guide = DoGuideAttentionLayer(batch_size, text_seq_len, text_hidden_dim, image_block_num,
                                              image_hidden_dim)

    def forward(self, text_feature, image_feature, use_source=1):
        text_out, image_out = self.strategies.get(use_source, self.strategies[0])(text_feature, image_feature)
        return text_out, image_out

    def process_strategy_0(self, text_feature, image_feature):
        # 使用原生的文本或者图片特征
        # 文本处理
        text_out = self.do_guide(0, text_feature, image_feature)
        # 图片处理
        image_out = self.do_guide(1, text_feature, image_feature)
        return text_out, image_out

    def process_strategy_1(self, text_feature, image_feature):
        # 使用文本引导图片的结果作为图片和文本中的文本特征
        text_out = self.do_guide(0, text_feature, image_feature)
        # 图片处理
        image_out = self.do_guide(1, text_out, image_feature)
        return text_out, image_out

    def process_strategy_2(self, text_feature, image_feature):
        # 使用图片引导文本的结果作为图片和文本中的图片特征
        image_out = self.do_guide(1, text_feature, image_feature)
        # 使用文本引导图片的结果作为图片和文本中的文本特征
        text_out = self.do_guide(0, text_feature, image_out)
        return text_out, image_out

    def process_strategy_3(self, text_feature, image_feature):
        # 文本和图片都进行优化
        image_out = self.do_guide(1, self.do_guide(0, text_feature, image_feature), image_feature)
        # 文本处理
        text_out = self.do_guide(0, text_feature, self.do_guide(1, text_feature, image_feature))
        return text_out, image_out

class LayerNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

        self.a = nn.Parameter(torch.ones(size))
        self.b = nn.Parameter(torch.zeros(size))

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a * (x - mean) / (std + self.eps) + self.b

class DoGuideAttentionLayer(nn.Module):

    def __init__(self, batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim):
        super(DoGuideAttentionLayer, self).__init__()
        self.batch_size = batch_size
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        # sparse attention
        self.text_sparse_attention = TextSparseAttention(text_seq_len=text_seq_len, text_hidden_dim=text_hidden_dim,
                                                         image_block_num=image_block_num,
                                                         image_hidden_dim=image_hidden_dim)
        self.image_sparse_attention = ImageSparseAttention(text_seq_len=text_seq_len, text_hidden_dim=text_hidden_dim,
                                                           image_block_num=image_block_num,
                                                           image_hidden_dim=image_hidden_dim)

        # MLP
        self.text_out = nn.Sequential(nn.Linear(text_hidden_dim, text_hidden_dim * 4), nn.ReLU(),
                                      nn.Linear(text_hidden_dim * 4, text_hidden_dim)).to(self.device)
        self.image_out = nn.Sequential(nn.Linear(image_hidden_dim, image_hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(image_hidden_dim * 4, image_hidden_dim)).to(self.device)

        # norm

        self.text_norm = LayerNorm(text_hidden_dim)
        self.image_norm = LayerNorm(image_hidden_dim)

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

    def forward(self, which_one, text_feature, image_feature):
        return self.process_text(text_feature, image_feature) if which_one == 0 else self.process_image(text_feature,
                                                                                                        image_feature)

    def process_text(self, text_feature, image_feature):
        # sparse attention
        out = self.text_sparse_attention(text_feature, image_feature)
        # norm
        out = self.text_norm(text_feature + out)
        # MLP out
        out = self.text_out(out)
        # norm
        out = self.text_norm(text_feature + out)

        return out

    def process_image(self, text_feature, image_feature):
        # sparse attention
        out = self.image_sparse_attention(text_feature, image_feature)
        # norm
        out = self.image_norm(image_feature + out)
        # MLP out
        out = self.image_out(out)
        # norm
        out = self.image_norm(image_feature + out)

        return out
