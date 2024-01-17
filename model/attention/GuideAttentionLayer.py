import torch
from torch import nn


from model.attention.ImageSparseAttention import ImageSparseAttention
from model.attention.TextSparseAttention import TextSparseAttention


# from GatedLayer import GatedLayer

'''
引导注意力层
输入特征f1，f2进行两次执行注意力运算
self-attention -> norm(Add) -> f1:Q, f2:K,V（spare max）-> norm(Add) -> MLP 
self-attention -> norm(Add) -> f2:Q, f1:K,V（spare max）-> norm(Add) -> MLP 

最后返回二者结果

self.norm_after_bert = nn.LayerNorm(normalized_shape=[batch_size, input_max_length, bert_hidden_size],
                                            eps=1e-6,
                                            device=device)
'''


class GuideAttentionLayer(nn.Module):

    def __init__(self, batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim):
        super(GuideAttentionLayer, self).__init__()
        self.batch_size = batch_size

        # sparse attention
        self.text_sparse_attention = TextSparseAttention(text_seq_len=text_seq_len, text_hidden_dim=text_hidden_dim,
                                                         image_block_num=image_block_num,
                                                         image_hidden_dim=image_hidden_dim)
        self.image_sparse_attention = ImageSparseAttention(text_seq_len=text_seq_len, text_hidden_dim=text_hidden_dim,
                                                           image_block_num=image_block_num,
                                                           image_hidden_dim=image_hidden_dim)

        # MLP
        self.text_out = nn.Sequential(nn.Linear(text_hidden_dim, text_hidden_dim * 4), nn.ReLU(),
                                      nn.Linear(text_hidden_dim * 4, text_hidden_dim))
        self.image_out = nn.Sequential(nn.Linear(image_hidden_dim, image_hidden_dim * 4), nn.ReLU(),
                                       nn.Linear(image_hidden_dim * 4, image_hidden_dim))

        # norm
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
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
        image_out = self.image_sparse_attention(image_feature, text_feature)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)
        # 2. MLP
        image_out = self.image_out(image_out)
        # norm
        image_out = self.norm(image_feature, image_out, self.image_norm)

        # 再对文本处理
        # 1. sparse attention
        text_out = self.text_sparse_attention(text_feature, image_out)
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
