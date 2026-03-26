# 读取标识符文件并返回标识符列表
import os

import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from tqdm import tqdm


def read_ids(file_path):
    with open(file_path, 'r') as f:
        return [line.strip().split('.')[0] for line in f.readlines()]


# 读取原始数据文件并返回所有数据
def read_data(file_path):
    with open(file_path, 'r') as f:
        return [eval(line.strip()) for line in f.readlines()]  # 使用 eval 解析列表


# 写入新文件
def write_data(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')


def allocate_att():
    # 读取标识符文件
    train_ids = read_ids('train_id.txt')
    valid_ids = read_ids('valid_id.txt')
    test_ids = read_ids('test_id.txt')

    # 读取原始数据文件
    raw_data = read_data('data.txt')

    # 分类数据并去除标识符
    train_data = [
        item[1:] if str(item[0]) in train_ids else item  # 去除标识符
        for item in raw_data
        if str(item[0]) in train_ids
    ]

    valid_data = [
        item[1:] if str(item[0]) in valid_ids else item  # 去除标识符
        for item in raw_data
        if str(item[0]) in valid_ids
    ]

    test_data = [
        item[1:] if str(item[0]) in test_ids else item  # 去除标识符
        for item in raw_data
        if str(item[0]) in test_ids
    ]

    # 如果列表为空，则添加默认值 "thing"
    train_data = [item if item else ["thing"] for item in train_data]
    valid_data = [item if item else ["thing"] for item in valid_data]
    test_data = [item if item else ["thing"] for item in test_data]

    # 将分类结果写入新文件
    write_data('att/train_att.txt', train_data)
    write_data('att/valid_att.txt', valid_data)
    write_data('att/test_att.txt', test_data)

    print("数据分类完成！")


def image_process():
    img_dir = '/Users/rayss/Public/读研经历/论文/dataset/bully/bully_data/'
    save_path = '/Users/rayss/Public/读研经历/论文/dataset/bully/tensor/'
    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for root, dirs, files in tqdm(os.walk(img_dir)):
        for filename in files:
            img_path = os.path.join(
                img_dir, filename
            )
            try:
                img = Image.open(img_path)
                img = img.convert('RGB')  # convert grey picture
                trainsform_img = transform(img)
                # image_tensor[mode].append(trainsform_img.unsqueeze(0))
                np.save(save_path + str(filename.split('.')[0]) + '.npy', trainsform_img.numpy())
            except Exception:
                pass


def transform_yes_no(file_path):
    # 文件路径

    # 读取并处理文件
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # 替换内容
    for i in range(len(lines)):
        lines[i] = '1\n' if lines[i].strip().lower() == 'yes' else '0\n' if lines[i].strip().lower() == 'no' else lines[
            i]

    # 覆盖原文件
    with open(file_path, 'w') as file:
        file.writelines(lines)

    print(f"文件 '{file_path}' 已更新")



if __name__ == "__main__":
    # 4640
    # 581
    # 580
    allocate_att()
    # image_process()
    # transform_yes_no('test_label.txt')
    # transform_yes_no('valid_label.txt')
    # transform_yes_no('train_label.txt')