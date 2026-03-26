import json
import pickle
import torch


def load_file(filename):
    with open(filename, 'rb') as filehandle:
        ret = pickle.load(filehandle)
        return ret

def load_2_file(filename):
    text_list = []
    with open(filename, 'r', encoding='utf-8') as file:
        data_list = json.load(file)
        for data in data_list:
            text_list.append(data['text'])
    return text_list

def load_bully_file(filename):

    # 读取文件并将每行作为列表元素
    with open(filename, 'r') as file:
        data_list = file.readlines()

    # 去除每行末尾的换行符（如果需要）
    return [line.strip() for line in data_list]
def load_att_file(att_file_path, mould):
    att_mould_list = []
    with open(att_file_path) as f:
        for att in f:
            att = eval(att)
            # 创建一个动态的字典来存储变量
            # 创建一个字典
            att_dict = {f'att_{i}': att[i] for i in range(len(att))}
            # 更新字典，确保每个位置都有默认值
            att_dict.update({f'att_{i}': '' for i in range(len(att), 5)})
            # att_mould_list.append(mould.format(att_0=att[0], att_1=att[1], att_2=att[2], att_3=att[3], att_4=att[4]))
            att_mould_list.append(mould.format(**att_dict).replace(", and", "").replace(", ,", ",").strip())
    return att_mould_list
class loader_text:
    def __init__(self):
        self.name="text"
        self.require=["tokenizer_roberta"]

    def prepare(self,input,opt):
        if "mould" not in opt:
            opt["mould"] = "A photo containing the {att_0}, {att_1}, {att_2}, {att_3} and {att_4}"
        source = opt.get("source", "bully")
        if source == "prepared":
            self.text = {
                "train": load_file(opt["data_path"] + "train_text"),
                "test": load_file(opt["data_path"] + "test_text"),
                "valid": load_file(opt["data_path"] + "valid_text")
            }
            att_base = opt.get("att_file_path", opt["data_path"] + "att/")
            self.att = {
                "train": load_att_file(att_base + "train_att.txt", opt["mould"]),
                "test": load_att_file(att_base + "test_att.txt", opt["mould"]),
                "valid": load_att_file(att_base + "valid_att.txt", opt["mould"])
            }
        elif source == "msd2":
            self.text = {
                "train": load_2_file(opt["data_2_path"] + "train.json"),
                "test": load_2_file(opt["data_2_path"] + "test.json"),
                "valid": load_2_file(opt["data_2_path"] + "valid.json")
            }
            # MSD2 without att files: use empty template text for compatibility.
            self.att = {mode: [""] * len(self.text[mode]) for mode in ["train", "test", "valid"]}
        elif source == "bully":
            self.text = {
                "train": load_bully_file(opt["data_bully_path"] + "train_text.txt"),
                "test": load_bully_file(opt["data_bully_path"] + "test_text.txt"),
                "valid": load_bully_file(opt["data_bully_path"] + "valid_text.txt")
            }
            self.att = {
                "train": load_att_file(opt["att_file_path"] + "train_att.txt", opt["mould"]),
                "test": load_att_file(opt["att_file_path"] + "test_att.txt", opt["mould"]),
                "valid": load_att_file(opt["att_file_path"] + "valid_att.txt", opt["mould"])
            }
        else:
            raise ValueError(f"Unsupported text loader source: {source}")
        if "len" not in opt:
            opt["len"]=100
        self.len=opt["len"]
        if "pad" not in opt:
            opt["pad"]=1
        self.pad=opt["pad"]
        self.tokenizer=input[list(input.keys())[0]]
        # self.tokenizer = SimpleTokenizer()
        # self.processor = CLIPProcessor.from_pretrained("/Users/rayss/pythonProjects/pretrained_model/clip-vit-base-patch32")

        self.text_mask = {
            "train":[],
            "test":[],
            "valid":[]
        }
        self.text_id = {
            "train":[],
            "test":[],
            "valid":[]     
        }

        self.att_id = {
            "train": [],
            "test": [],
            "valid": []
        }
        self.att_mask = {
            "train": [],
            "test": [],
            "valid": []
        }
        # self.extract_feature_by_processor(self.text, self.text_mask, self.text_id)
        # self.extract_feature_by_processor(self.att, self.att_mask, self.att_id)
        self.extract_feature_by_tokenizer(self.text, self.text_mask, self.text_id)
        self.extract_feature_by_tokenizer(self.att, self.att_mask, self.att_id)

    def extract_feature_by_tokenizer(self, source_data, mask, id):
        for mode in source_data.keys():
            for index, text in enumerate(source_data[mode]):
                indexed_tokens_for_text = self.tokenizer(text)['input_ids']
                # indexed_tokens_for_text = self.tokenizer(text)
                if len(indexed_tokens_for_text) > self.len:
                    indexed_tokens_for_text = indexed_tokens_for_text[0:self.len]
                text_mask = torch.BoolTensor(
                    [0] * len(indexed_tokens_for_text) + [1] * (self.len - len(indexed_tokens_for_text)))
                indexed_tokens_for_text += [self.pad] * (self.len - len(indexed_tokens_for_text))
                text_id = torch.tensor(indexed_tokens_for_text)
                mask[mode].append(text_mask)
                id[mode].append(text_id)

    def extract_feature_by_processor(self, source_data, mask, id):
        from transformers import CLIPProcessor
        if not hasattr(self, "processor"):
            raise RuntimeError("processor is not initialized before extract_feature_by_processor")
        for mode in source_data.keys():
            for index, text in enumerate(source_data[mode]):
                indexed_tokens_for_text = self.processor(text=text).data['input_ids']
                if len(indexed_tokens_for_text) > self.len:
                    indexed_tokens_for_text = indexed_tokens_for_text[0:self.len]
                text_mask = torch.BoolTensor(
                    [0] * len(indexed_tokens_for_text) + [1] * (self.len - len(indexed_tokens_for_text)))
                indexed_tokens_for_text += [self.pad] * (self.len - len(indexed_tokens_for_text))
                text_id = torch.tensor(indexed_tokens_for_text)
                mask[mode].append(text_mask)
                id[mode].append(text_id)

    def get(self,result,mode,index):
        result["text_mask"]= self.text_mask[mode][index]
        result["text"]=self.text_id[mode][index]
        result["att_mask"] = self.att_mask[mode][index]
        result["att"] = self.att_id[mode][index]


    def getlength(self,mode):
        return len(self.text[mode])
