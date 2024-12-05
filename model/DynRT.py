import argparse

import torch
import timm
import model
from transformers import RobertaModel, CLIPModel

from model.TRAR.cls_layer import cls_layer_both
from model.attention.BridgeInfoLayer import BridgeInfoLayer
from model.attention.CrossModalTransformerLayer import CrossModalTransformerLayer
from model.attention.GuideAttentionLayer import GuideAttentionLayer
from model.attention.MultimodalFusionLayer import MultimodalFusionLayer
from model.attention.TraditionalAttentionLayer import TraditionalAttentionLayer

from model.mamba_clip_main.main import get_model, get_args_parser

def freeze_layers(model):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

class DynRT(torch.nn.Module):
  # define model elements
    def __init__(self,bertl_text,vit, opt,batch_size=32):
        super(DynRT, self).__init__()

        # clip_model = CLIPModel.from_pretrained("/Users/rayss/pythonProjects/pretrained_model/clip-vit-base-patch32")
        # self.clip_image = clip_model.vision_model
        # self.clip_text = clip_model.text_model
        # self.fc_clip = torch.nn.Linear(512, 768)
        # parser = argparse.ArgumentParser('A-CLIP training and evaluation', parents=[get_args_parser()])
        # args = parser.parse_args()
        # self.clip_mamba = get_model(args)
        self.bertl_text = bertl_text
        self.opt = opt
        self.vit = vit
        self.bridge_info_layer = BridgeInfoLayer(opt)
        self.cross_modal_transformer_layer = CrossModalTransformerLayer(opt)
        self.guide_attention_layer = GuideAttentionLayer(batch_size=batch_size, text_seq_len=opt['len'],
                                                         text_hidden_dim=opt['mlp_size'],
                                                         image_block_num=opt['IMG_SCALE'] * opt['IMG_SCALE'],
                                                         image_hidden_dim=opt['mlp_size'], use_type=3, use_source=1)
        self.tradition_attention_layer = TraditionalAttentionLayer(text_seq_len=opt['len'],
                                                                   text_hidden_dim=opt['mlp_size'],
                                                                   image_block_num=opt['IMG_SCALE'] * opt['IMG_SCALE'],
                                                                   image_hidden_dim=opt['mlp_size'], use_type=3)
        if not self.opt["finetune"]:
            freeze_layers(self.bertl_text)
            freeze_layers(self.vit)
        assert("input1" in opt)
        assert("input2" in opt)
        assert("input3" in opt)
        self.input1=opt["input1"]
        self.input2=opt["input2"]
        self.input3=opt["input3"]
        self.input4=opt["input4"]
        self.input5=opt["input5"]

        self.trar = model.TRAR.DynRT(opt)
        self.cls_layer = cls_layer_both(opt["hidden_size"], opt["output_size"])
        self.sigm = torch.nn.Sigmoid()
        self.fusion_layer = MultimodalFusionLayer(opt)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(opt['dropout']),
            torch.nn.Linear(opt["output_size"],2)
        )

    def bert_forward(self,x, mask):
        # 如果roberta模型，则走bert_forward，否则就走自己的模型结果
        if self.opt['model'] == 'model_roberta':
            # (bs, max_len, dim)
            bert_embed_text = self.bertl_text.embeddings(input_ids=x)
            # (bs, max_len, dim)
            # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
            for i in range(self.opt["roberta_layer"]):
                bert_embed_text = self.bertl_text.encoder.layer[i](bert_embed_text)[0]
        else:
            bert_embed_text = self.bertl_text(input_ids=x,attention_mask=mask)['last_hidden_state']

        return bert_embed_text

    def vit_forward(self,x):
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        return x[:,1:]

    def clip_text_forward(self, x):
        clip_embed_text = self.clip_text.embeddings(x)
        # (bs, max_len, dim)
        # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        for i in range(6):
            clip_embed_text = self.clip_text.encoder.layers[i](clip_embed_text, None, None)[0]
        return self.fc_clip(clip_embed_text)

    def clip_image_forward(self, x):
        clip_embed_image = self.clip_image.embeddings(x)
        # (bs, max_len, dim)
        # bert_text = self.bertl_text.encoder.layer[0](bert_embed_text)[0]
        for i in range(6):
            clip_embed_image = self.clip_image.encoder.layers[i](clip_embed_image, None, None)[0]
        return clip_embed_image[:, 1:]

    # forward propagate input
    def forward(self, input):
        # 属性
        bert_embed_att = self.bert_forward(input[self.input4], input[self.input5])
        # bert_embed_att = self.clip_text_forward(input[self.input4])
        # 文本
        bert_embed_text = self.bert_forward(input[self.input1], input[self.input3])
        # bert_embed_text = self.clip_text_forward(input[self.input1])
        # 图像 (bs, grid_num, dim)
        img_feat = self.vit_forward(input[self.input2])

        # res = self.clip_mamba(input[self.input2], input[self.input4])
        # bert_embed_text = res['text_embed']
        # img_feat = res['image_embed']
        # img_feat = self.clip_image_forward(input[self.input2])

        # 属性关联
        text_incongruity, image_incongruity = self.bridge_info_layer(bert_embed_text, input[self.input3],
                                                                     bert_embed_att,
                                                                     input[self.input5], img_feat)
        text_incongruity, image_incongruity = self.cross_modal_transformer_layer(text_incongruity, image_incongruity)
        # 引导
        # bert_embed_text, img_feat = self.guide_attention_layer(text_att, img_att)
        # bert_embed_text, img_feat = self.tradition_attention_layer.process(bert_embed_text, img_feat)

        # (out1, lang_emb, img_emb) = self.trar(img_feat, bert_embed_text,input[self.input3].unsqueeze(1).unsqueeze(2))

        out = self.fusion_layer(text_incongruity, image_incongruity, bert_embed_text, img_feat)
        out = self.classifier(out)
        out = self.sigm(out)
        del bert_embed_text, img_feat

        return out, text_incongruity, image_incongruity, bert_embed_att


def build_DynRT(opt,requirements):
    bertl_text = get_text_encoder(opt)
    
    # bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
    if "vitmodel" not in opt:
        opt["vitmodel"] = "vit_base_patch32_224"
    # vit = timm.create_model(opt["vitmodel"], pretrained=True)
    pretrained_cfg = timm.models.create_model(opt["vitmodel"]).default_cfg
    pretrained_cfg['file'] = r'/Users/rayss/pythonProjects/pretrained_model/vit/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'
    vit = timm.models.create_model(opt["vitmodel"], pretrained=True, pretrained_cfg=pretrained_cfg)

    return DynRT(bertl_text, vit, opt,requirements['batch_size'])


def get_text_encoder(opt):
    def get_roberta(path):
        from transformers import RobertaModel
        return RobertaModel.from_pretrained(path)

    def get_bert(path):
        from transformers import BertModel
        return BertModel.from_pretrained(path)

    def get_albert(path):
        from transformers import AlbertModel
        return AlbertModel.from_pretrained(path)

    def get_xlnet(path):
        from transformers import XLNetModel
        return XLNetModel.from_pretrained(path)

    model = {
        "model_roberta": get_roberta,
        "model_bert": get_bert,
        "model_albert": get_albert,
        "model_xlnet": get_xlnet
    }
    return model[opt['model']](opt[opt['model']])
