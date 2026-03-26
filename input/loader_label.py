import json
import pickle

def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

def load_2_file(filename):
    label_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
        for data in data_list:
            label_list.append(data['label'])
    return label_list

def load_bully_file(filename):

    # 读取文件并将每行作为列表元素
    with open(filename, 'r') as file:
        data_list = file.readlines()

    # 去除每行末尾的换行符（如果需要）
    return [line.strip() for line in data_list]
class loader_label:
    def __init__(self):
        self.name="label"
        self.require=[]

    def prepare(self,input,opt):
        source = opt.get("source", "bully")
        if opt["test_label"]:
            if source == "prepared":
                self.label = {
                    "train": load_file(opt["data_path"] + "train_labels"),
                    "test": load_file(opt["data_path"] + "test_labels"),
                    "valid": load_file(opt["data_path"] + "valid_labels")
                }
            elif source == "msd2":
                self.label = {
                    "train": load_2_file(opt["data_2_path"] + "train.json"),
                    "test": load_2_file(opt["data_2_path"] + "test.json"),
                    "valid": load_2_file(opt["data_2_path"] + "valid.json")
                }
            elif source == "bully":
                self.label = {
                    "train": load_bully_file(opt["data_bully_path"] + "train_label.txt"),
                    "test": load_bully_file(opt["data_bully_path"] + "test_label.txt"),
                    "valid": load_bully_file(opt["data_bully_path"] + "valid_label.txt")
                }
            else:
                raise ValueError(f"Unsupported label loader source: {source}")

        else:
            self.label ={
                "train":load_file(opt["data_path"] + "train_labels"),
                "valid":load_file(opt["data_path"] + "valid_labels")
            }

    def get(self,result,mode,index):
        result["label"]=int(self.label[mode][index])
        # result["index"]=index

    def getlength(self,mode):
        return len(self.label[mode])
