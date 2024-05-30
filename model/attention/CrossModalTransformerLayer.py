import torch.nn
from torch import nn

from model.TRAR import LayerNorm
from model.TRAR.trar import FFN
from model.attention.GuideAttention import GuideAttention
from model.attention.MHAtt import MHAtt


class CrossModalTransformerLayer(torch.nn.Module):
    def __init__(self, opt, batch_size=32):
        super(CrossModalTransformerLayer, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.seq_len = opt['len']
        self.block_num = opt['IMG_SCALE'] * opt['IMG_SCALE']
        self.fc = nn.Linear(self.seq_len+self.block_num, self.seq_len)
        # batch_size, text_seq_len, text_hidden_dim, image_block_num, image_hidden_dim, use_source
        self.cross_attention = GuideAttention(batch_size, self.seq_len, opt["hidden_size"], self.block_num, opt["hidden_size"],
                                              use_source=2).to(self.device)
        # self.text_cross_attn = MHAtt(opt["hidden_size"], opt["hidden_size"])
        # self.image_cross_attn = MHAtt(opt["hidden_size"], opt["hidden_size"])

        self.text_self_attn = MHAtt(opt["hidden_size"], opt["hidden_size"])
        self.image_self_attn = MHAtt(opt["hidden_size"], opt["hidden_size"])

        self.text_ffn = FFN(opt)
        self.image_ffn = FFN(opt)

        self.dropout1 = nn.Dropout(opt["dropout"])
        self.norm1 = LayerNorm(opt["hidden_size"])

        self.dropout2 = nn.Dropout(opt["dropout"])
        self.norm2 = LayerNorm(opt["hidden_size"])

        self.dropout3 = nn.Dropout(opt["dropout"])
        self.norm3 = LayerNorm(opt["hidden_size"])

    def forward(self, text_feature, image_feature):
        # 交叉
        text_feature, image_feature = self.cross_attention(text_feature, image_feature)

        # 图像线路
        # image_feature = self.norm1(image_feature + self.dropout1(
        #     self.text_cross_attn(image_feature, text_feature, text_feature)
        # ))  # (64, 49, 512) # (bs, 49, 768)

        image_feature = self.norm2(image_feature + self.dropout2(
            self.image_self_attn(v=image_feature, k=image_feature, q=image_feature)
        ))

        image_feature = self.norm3(image_feature + self.dropout3(
            self.image_ffn(image_feature)
        ))
        # 文本线路
        # text_feature = self.norm1(text_feature + self.dropout1(
        #     self.text_cross_attn(text_feature, image_feature, image_feature)
        # ))  # (64, 49, 512) # (bs, 49, 768)

        text_feature = self.norm2(text_feature + self.dropout2(
            self.text_self_attn(v=text_feature, k=text_feature, q=text_feature)
        ))

        text_feature = self.norm3(text_feature + self.dropout3(
            self.text_ffn(text_feature)
        ))

        return text_feature, image_feature
