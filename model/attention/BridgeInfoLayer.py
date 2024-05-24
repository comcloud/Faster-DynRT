'''
桥梁信息层
主要用来构建文本和图像属性以及图像与图像属性的之间构造关系
'''
import math

import torch.nn
import torch.nn.functional as F
from torch import nn

from model.TRAR import LayerNorm
from model.TRAR.trar import LSAM_ED, FFN
from model.attention.MHAtt import MHAtt


class BridgeInfoLayer(torch.nn.Module):
    def __init__(self, opt):
        super(BridgeInfoLayer, self).__init__()
        self.dynamic_net = LSAM_ED(opt)
        self.biaff_trans = BiaffineTransformer(opt)

    def forward(self, text_feature, text_mask, att_feature, att_mask, image_feature):

        # 文属
        text_feature, _ = self.dynamic_net(
            text_feature,
            att_feature,
            text_mask.unsqueeze(1).unsqueeze(2),
            att_mask.unsqueeze(1).unsqueeze(2)
        )

        image_feature = self.biaff_trans(att_feature, image_feature)

        return text_feature, image_feature


class BiaffineTransformer(nn.Module):
    def __init__(self, opt):
        super(BiaffineTransformer, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.seq_len = opt['len']
        self.block_num = opt['IMG_SCALE'] * opt['IMG_SCALE']
        self.text_biaffine = Biaffine(self.seq_len, self.seq_len, self.seq_len).to(self.device)
        self.image_biaffine = Biaffine(self.block_num, self.seq_len, self.block_num).to(self.device)

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

    def forward(self, att_feature, image_feature):

        image_feature = self.norm1(image_feature + self.dropout1(
            self.image_biaffine(image_feature, att_feature)
        ))  # (64, 49, 512) # (bs, 49, 768)

        image_feature = self.norm2(image_feature + self.dropout2(
            self.image_self_attn(v=image_feature, k=image_feature, q=image_feature)
        ))

        image_feature = self.norm3(image_feature + self.dropout3(
            self.image_ffn(image_feature)
        ))
        return image_feature


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

        d_k = x.size(-1)
        bilinar_mapping = torch.bmm(x, y.transpose(-2, -1))
        bilinar_mapping = F.softmax(bilinar_mapping, dim=-1) / math.sqrt(d_k)
        bilinar_mapping = torch.bmm(bilinar_mapping, z)
        return bilinar_mapping



if __name__ == '__main__':
    t_f = torch.randn(2, 100, 768)
    a_f = torch.randn(2, 100, 768)
    i_f = torch.randn(2, 49, 768)

    # biaffine = Biaffine(49, 100, 49)
    # res = biaffine(i_f, a_f)
    # print(res.size())
    opt = {
        "len": 100,
        "IMG_SCALE": 7,
        "hidden_size": 768,
        "dropout": 0.5,
        "ffn_size": 768
    }
    model = BridgeInfoLayer(opt)
    t_f, i_f = model(t_f, None, a_f, None, i_f)
    t_f = torch.mean(t_f, dim=1)
    i_f = torch.mean(i_f, dim=1)
    print(t_f.size())
    print(i_f.size())
