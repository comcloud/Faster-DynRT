import random

# 文件路径
file_paths = {
    "train": {"id": "train_id.txt", "label": "train_label.txt", "text": "train_text.txt"},
    "valid": {"id": "valid_id.txt", "label": "valid_label.txt", "text": "valid_text.txt"},
    "test": {"id": "test_id.txt", "label": "test_label.txt", "text": "test_text.txt"}
}


# 读取数据函数
def read_data(file_paths):
    data = {}
    for dataset in file_paths:
        data[dataset] = {
            "id": open(file_paths[dataset]["id"]).readlines(),
            "label": open(file_paths[dataset]["label"]).readlines(),
            "text": open(file_paths[dataset]["text"]).readlines()
        }
    return data


# 重新分配数据
def reallocate_data(data, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    # 组合所有数据
    all_data = {
        "id": [],
        "label": [],
        "text": []
    }

    for dataset in data:
        all_data["id"] += data[dataset]["id"]
        all_data["label"] += data[dataset]["label"]
        all_data["text"] += data[dataset]["text"]

    # 将数据按标签分开
    pos_data = {"id": [], "label": [], "text": []}
    neg_data = {"id": [], "label": [], "text": []}

    for i in range(len(all_data["label"])):
        if all_data["label"][i].strip() == "1":
            pos_data["id"].append(all_data["id"][i])
            pos_data["label"].append(all_data["label"][i])
            pos_data["text"].append(all_data["text"][i])
        else:
            neg_data["id"].append(all_data["id"][i])
            neg_data["label"].append(all_data["label"][i])
            neg_data["text"].append(all_data["text"][i])

    # 打乱数据
    random.shuffle(pos_data["id"])
    random.shuffle(neg_data["id"])

    # 合并数据并分配
    total_pos = len(pos_data["id"])
    total_neg = len(neg_data["id"])

    # 分配样本数量
    train_pos = int(total_pos * train_ratio)
    valid_pos = int(total_pos * valid_ratio)
    test_pos = total_pos - train_pos - valid_pos

    train_neg = int(total_neg * train_ratio)
    valid_neg = int(total_neg * valid_ratio)
    test_neg = total_neg - train_neg - valid_neg

    # 创建新的数据集
    train_data = {
        "id": pos_data["id"][:train_pos] + neg_data["id"][:train_neg],
        "label": pos_data["label"][:train_pos] + neg_data["label"][:train_neg],
        "text": pos_data["text"][:train_pos] + neg_data["text"][:train_neg],
    }

    valid_data = {
        "id": pos_data["id"][train_pos:train_pos + valid_pos] + neg_data["id"][train_neg:train_neg + valid_neg],
        "label": pos_data["label"][train_pos:train_pos + valid_pos] + neg_data["label"][
                                                                      train_neg:train_neg + valid_neg],
        "text": pos_data["text"][train_pos:train_pos + valid_pos] + neg_data["text"][train_neg:train_neg + valid_neg],
    }

    test_data = {
        "id": pos_data["id"][train_pos + valid_pos:] + neg_data["id"][train_neg + valid_neg:],
        "label": pos_data["label"][train_pos + valid_pos:] + neg_data["label"][train_neg + valid_neg:],
        "text": pos_data["text"][train_pos + valid_pos:] + neg_data["text"][train_neg + valid_neg:],
    }

    return train_data, valid_data, test_data


# 将数据写回文件
def write_data(data, prefix):
    with open(f"{prefix}_id.txt", "w") as f_id, \
            open(f"{prefix}_label.txt", "w") as f_label, \
            open(f"{prefix}_text.txt", "w") as f_text:
        for i in range(len(data["id"])):
            f_id.write(data["id"][i])
            f_label.write(data["label"][i])
            f_text.write(data["text"][i])


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


def read_ids(file_path):
    with open(file_path, 'r') as f:
        return [line.strip().split('.')[0] for line in f.readlines()]


# 读取原始数据文件并返回所有数据
def read_att_data(file_path):
    with open(file_path, 'r') as f:
        return [eval(line.strip()) for line in f.readlines()]  # 使用 eval 解析列表


# 写入新文件
def write_att_data(file_path, data):
    with open(file_path, 'w') as f:
        for item in data:
            f.write(str(item) + '\n')


def allocate_att():
    # 读取标识符文件
    train_ids = read_ids('train_id.txt')
    valid_ids = read_ids('valid_id.txt')
    test_ids = read_ids('test_id.txt')

    # read_att_data
    raw_data = read_att_data('data.txt')

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
    write_att_data('att/train_att.txt', train_data)
    write_att_data('att/valid_att.txt', valid_data)
    write_att_data('att/test_att.txt', test_data)

    print("数据分类完成！")


# 主流程
def main():
    '''
    读取数据，随机打乱，并写出train,valid,test
    '''
    # 读取数据
    data = read_data(file_paths)

    # 重新分配数据
    train_data, valid_data, test_data = reallocate_data(data)

    # 写回文件
    write_data(train_data, "train")
    write_data(valid_data, "valid")
    write_data(test_data, "test")

    '''
        label变为1，0
    '''
    transform_yes_no('test_label.txt')
    transform_yes_no('valid_label.txt')
    transform_yes_no('train_label.txt')

    '''
        分配属性
    '''
    allocate_att()


if __name__ == "__main__":
    main()
