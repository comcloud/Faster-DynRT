import os
import pickle

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torchvision import transforms

from util import center_crop_img, GradCAM, show_cam_on_image


def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret


class ReshapeTransform:
    def __init__(self, model, input_size=(224, 224), patch_size=(32, 32)):
        # input_size = model.patch_embed.img_size
        # patch_size = model.patch_embed.patch_size
        self.h = input_size[0] // patch_size[0]
        self.w = input_size[1] // patch_size[1]

    def __call__(self, x):
        # remove cls token and reshape
        # [batch_size, num_tokens, token_dim]
        # result = x[:, 1:, :].reshape(x.size(0),
        #                              self.h,
        #                              self.w,
        #                              x.size(2))
        result = x.reshape(x.size(0),
                           self.h,
                           self.w,
                           x.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        # [batch_size, H, W, C] -> [batch, C, H, W]
        result = result.permute(0, 3, 1, 2)
        return result


def gen_image_heat_map(model, tokenizer_roberta):
    # target_layers = [self.model.vit.blocks[-1].norm1]
    target_layers = [model.cross_modal_transformer_layer.cross_attention.strategy.do_guide.image_norm]
    # load image
    # image_root = r"/Users/rayss/Public/读研经历/论文/ironyDetection/imageVector2/"
    image_root = r"/Users/rayss/Public/读研经历/论文/ironyDetection/imageVector/"
    files = os.listdir(image_root)
    train_ids = load_file("input/prepared_clean/train_id")
    valid_ids = load_file("input/prepared_clean/valid_id")
    test_ids = load_file("input/prepared_clean/test_id")
    train_texts = load_file("input/prepared_clean/train_text")
    valid_texts = load_file("input/prepared_clean/valid_text")
    test_texts = load_file("input/prepared_clean/test_text")

    def get_att(att_file_path="/Users/rayss/pythonProjects/DynRT/checkpoint/extract_all",
                mould='A photo containing the {att_0}, {att_1}, {att_2}, {att_3} and {att_4}'):
        att_mould_dict = {}
        with open(att_file_path) as f:
            for att in f:
                att = eval(att)
                att_mould_dict[int(att[0])] = mould.format(att_0=att[1], att_1=att[2], att_2=att[3], att_3=att[4],
                                                           att_4=att[5])
        return att_mould_dict

    for i in range(0, 1):
        # for i in range(len(files)):
        img_path = image_root + files[i]
        do_gen_image_heat_map(model, target_layers, img_path, files[i], train_ids, train_texts, valid_ids, valid_texts,
                              test_ids,
                              test_texts, get_att(), tokenizer_roberta)


def do_gen_image_heat_map(model, target_layers, img_path, file_name, train_ids, train_texts, valid_ids, valid_texts,
                          test_ids,
                          test_texts, att_dict, tokenizer_roberta):
    # model = vit_base_patch16_224()
    # 链接: https://pan.baidu.com/s/1zqb08naP0RPqqfSXfkB2EA  密码: eu9f
    # weights_path = "./vit_base_patch16_224.pth"
    # model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    # Since the final classification is done on the class token computed in the last attention block,
    # the output will not be affected by the 14x14 channels in the last layer.
    # The gradient of the output with respect to them, will be 0!
    # We should chose any layer before the final attention block.
    # 读取文本：截断file_name获取text_id，从train_id,test_id,valid_id中查询，得到其中的位置索引；根据此索引从对应的text中获取
    text_id = file_name.split('.')[0]
    try:
        if text_id in train_ids:
            idx = train_ids.index(text_id)
            text = train_texts[idx]
        elif text_id in valid_ids:
            idx = valid_ids.index(text_id)
            text = valid_texts[idx]
        else:
            idx = test_ids.index(text_id)
            text = test_texts[idx]
    except:
        return
    text_mask, text_tensor = get_text_tensor(text, tokenizer_roberta)
    att_mask, att_tensor = get_text_tensor(att_dict[int(text_id)], tokenizer_roberta)

    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = np.array(img, dtype=np.uint8)
    img = center_crop_img(img, 224)
    # [C, H, W]
    img_tensor = data_transform(img)
    # expand batch dimension
    # [C, H, W] -> [N, C, H, W]
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    cam = GradCAM(model=model,
                  target_layers=target_layers,
                  use_cuda=False,
                  reshape_transform=ReshapeTransform(model))
    try:
        grayscale_cam = cam(
            inputs={"text_mask": text_mask, "text": text_tensor, "att_mask": att_mask, "att": att_tensor,
                    "img": img_tensor})
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(visualization)
        plt.savefig("/Users/rayss/Public/读研经历/论文/ironyDetection/ACM英文论文/heat_map/" + file_name, dpi=300)
        # plt.savefig("/Users/rayss/pythonProjects/DynRT/checkpoint/new/heat_map/" + file_name, dpi=300)
    except:
        pass


def get_text_tensor(text, tokenizer_roberta):
    text_tensor = tokenizer_roberta(text)['input_ids']
    if len(text_tensor) > 100:
        text_tensor = text_tensor[0:100]
    text_mask = torch.BoolTensor(
        [0] * len(text_tensor) + [1] * (100 - len(text_tensor))).unsqueeze(0)
    text_tensor += [1] * (100 - len(text_tensor))
    text_tensor = torch.tensor(text_tensor).unsqueeze(0)
    return text_mask, text_tensor


def visualize_and_save_tsne_2d_withgate(all_out, y_test, save_path):
    # all_out = []  # 用于存储所有批次的输出特征

    # 将所有输出堆叠成一个数组
    # size is (batch_size, 49, 768)
    all_out = np.vstack(all_out)

    # 对输出进行 t-SNE 降维
    tsne = TSNE(n_components=2, random_state=0)
    out_2d = tsne.fit_transform(all_out.reshape(all_out.shape[0], -1))

    # 定义不同类别对应的颜色
    label_to_color = {0: 'b', 1: 'lightgreen'}

    # 定义不同类别对应的标签
    label_to_text = {0: 'non-sarcasm', 1: 'sarcasm'}

    # 在二维空间中可视化样本
    y_test = np.concatenate(y_test)
    for label in np.unique(y_test):
        indices = np.where(y_test == label)
        plt.scatter(out_2d[indices, 0], out_2d[indices, 1], c=label_to_color[label], label=label_to_text[label])

    plt.legend()

    # 移除坐标轴刻度
    plt.xticks([])
    plt.yticks([])

    # 保存图形为PDF格式
    # plt.savefig(save_path, format='png', bbox_inches='tight', pad_inches=0, dpi=350)
    plt.show()


if __name__ == '__main__':
    all_feature = [torch.randn((32, 149, 768)).numpy()]
    all_labels = [torch.randint(low=0, high=2, size=(32, 1)).squeeze(1).numpy()]
    # if i > 20:
    #     input = {}
    #     for key in batch:
    #         input[key] = batch[key].to(self.device)
    #     scores, lang_feat, img_feat, bert_embed_att = self.model(input)
    #
    #     if i == 40:
    #         break
    # # 散点图
    # all_feature.append(torch.concat((lang_feat, img_feat, bert_embed_att), dim=1).numpy())
    # all_labels.append(input["label"].numpy())
    visualize_and_save_tsne_2d_withgate(all_feature, all_labels, '/Users/rayss/pythonProjects/DynRT/checkpoint/img.png')
