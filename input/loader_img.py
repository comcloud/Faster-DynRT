import pickle
import torch
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret


class loader_img:
    def __init__(self):
        self.name="img"
        self.require=[]

    def prepare(self,input,opt):
        self.id ={
            "train":load_file(opt["data_path"] + "train_id"),
            "test":load_file(opt["data_path"] + "test_id"),
            "valid":load_file(opt["data_path"] + "valid_id")
        }
        self.transform_image_path = opt["transform_image"]

    def get(self,result,mode,index):
        img_path=os.path.join(
                self.transform_image_path,
                "{}.npy".format(self.id[mode][index])
            )
        img = torch.from_numpy(np.load(img_path))
        result["img"]=img
    

    def getlength(self,mode):
        return len(self.id[mode])
    #
    # def get(self, result, mode, index):
    #     # 如果文件不存在，则直接进行新建
    #     # 图片所在目录
    #     file_path = '/Users/rayss/Public/读研经历/论文/ironyDetection/imageVector2/' + self.id[mode][index] + ".jpg"
    #     new_file_path = "/Users/rayss/pythonProjects/DynRT/input/image_tensor/" + self.id[mode][index] + ".npy"
    #     if not os.path.exists(new_file_path):
    #         # 从文件加载图像数据
    #         image = Image.open(file_path)
    #         transform = transforms.Compose([
    #             transforms.Resize(256),
    #             transforms.CenterCrop(224),
    #             # transforms.Grayscale(),
    #             # 将图像转换为黑白，则当前维度为(batch_size, height, width)，否则则是3通道：(batch_size, channels, height, width)
    #             transforms.ToTensor(),
    #             transforms.Normalize(mean=[0.485], std=[0.229]),  # 标准化黑白图像的均值和标准差
    #         ])
    #         image = transform(image)
    #         # 将图像转换为NumPy数组
    #         data = image.numpy()
    #         # 保存数组为NPY文件 'image_tensor/691699963584200705.npy'
    #         np.save(new_file_path, data)
    #     img_path = os.path.join(
    #         self.transform_image_path,
    #         "{}.npy".format(self.id[mode][index])
    #     )
    #     img = torch.from_numpy(np.load(img_path))
    #     result["img"]=img