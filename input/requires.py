

def get_tokenizer_roberta(results,opt):
    from transformers import RobertaTokenizer
    results["tokenizer_roberta"]=RobertaTokenizer.from_pretrained(opt['path'])

def get_tokenizer_bert(results,opt):
    from transformers import BertTokenizer
    results["tokenizer_bert"]=BertTokenizer.from_pretrained(opt['path'])

def get_tokenizer_albert(results,opt):
    from transformers import AlbertTokenizer
    results["tokenizer_albert"]=AlbertTokenizer.from_pretrained(opt['path'])

def get_tokenizer_xlnet(results,opt):
    from transformers import XLNetTokenizer
    results["tokenizer_xlnet"]=XLNetTokenizer.from_pretrained(opt['path'])


