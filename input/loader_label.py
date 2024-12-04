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
        if opt["test_label"]:
            self.label ={
                "train":load_file(opt["data_path"] + "train_labels"),
                "test":load_file(opt["data_path"] + "test_labels"),
                "valid":load_file(opt["data_path"] + "valid_labels")
            }
            # self.label2 = {
            #     "train": load_2_file(opt["data_2_path"] + "train.json"),
            #     "test": load_2_file(opt["data_2_path"] + "test.json"),
            #     "valid": load_2_file(opt["data_2_path"] + "valid.json")
            # }
            # for m in ['train', 'test', 'valid']:
            #     print('-----------' + m + '--------------')
            #     diff_indexes = list(filter(lambda i: self.label[m][i] != self.label2[m][i], range(len(self.label[m]))))
            #     diff_indexes = [i + 1 for i in diff_indexes]
            #     print(diff_indexes)
            # self.label = {
            #     "train": load_bully_file(opt["data_bully_path"] + "train_label.txt"),
            #     "test": load_bully_file(opt["data_bully_path"] + "test_label.txt"),
            #     "valid": load_bully_file(opt["data_bully_path"] + "valid_label.txt")
            # }

        else:
            self.label ={
                "train":load_file(opt["data_path"] + "train_labels"),
                "valid":load_file(opt["data_path"] + "valid_labels")
            }

    def get(self,result,mode,index):
        result["label"]=self.label[mode][index]
        # result["index"]=index

    def getlength(self,mode):
        return len(self.label[mode])