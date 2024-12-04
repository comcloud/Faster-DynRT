import json
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

def load_2_file(filename):
    id_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
        for data in data_list:
            id_list.append(data['image_id'])
    return id_list
def load_bully_file(filename):

    # 读取文件并将每行作为列表元素
    with open(filename, 'r') as file:
        data_list = file.readlines()

    # 去除每行末尾的换行符（如果需要）
    return [line.strip() for line in data_list]

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
        # self.id = {
        #     "train": load_2_file(opt["data_2_path"] + "train.json"),
        #     "test": load_2_file(opt["data_2_path"] + "test.json"),
        #     "valid": load_2_file(opt["data_2_path"] + "valid.json")
        # }
        # self.id = {
        #     "train": load_bully_file(opt["data_bully_path"] + "train_id.txt"),
        #     "test": load_bully_file(opt["data_bully_path"] + "test_id.txt"),
        #     "valid": load_bully_file(opt["data_bully_path"] + "valid_id.txt")
        # }
        self.transform_image_path = opt["transform_image"]

    # def get(self,result,mode,index):
    #     img_path=os.path.join(
    #             self.transform_image_path,
    #             "{}.npy".format(self.id[mode][index])
    #         )
    #     img = torch.from_numpy(np.load(img_path))
    #     result["img"]=img
    

    def getlength(self,mode):
        return len(self.id[mode])

    def get(self, result, mode, index):
        # 如果文件不存在，则直接进行新建
        # 图片所在目录
        file_path = '/Users/rayss/Public/读研经历/论文/ironyDetection/imageVector2/' + str(self.id[mode][index]) + ".jpg"
        new_file_path = "/Users/rayss/Public/读研经历/论文/ironyDetection/image_tensor_224/" + str(self.id[mode][index]) + ".npy"
        if not os.path.exists(new_file_path):
            # 从文件加载图像数据
            image = Image.open(file_path)
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                # transforms.Grayscale(),
                # 将图像转换为黑白，则当前维度为(batch_size, height, width)，否则则是3通道：(batch_size, channels, height, width)
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485], std=[0.229]),  # 标准化黑白图像的均值和标准差
            ])
            image = transform(image)
            # 将图像转换为NumPy数组
            data = image.numpy()
            # 保存数组为NPY文件 'image_tensor/691699963584200705.npy'
            np.save(new_file_path, data)
        # img_path = os.path.join(
        #     self.transform_image_path,
        #     "{}.npy".format(self.id[mode][index])
        # )
        img = torch.from_numpy(np.load(new_file_path))
        result["img"]=img