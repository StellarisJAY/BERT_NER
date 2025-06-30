from transformers import BertTokenizerFast
from model import NERModel
import torch   
import yaml
from dataset import get_label_dict
import os

def predict(model:NERModel, text:str, tokenizer: BertTokenizerFast, label_dict: dict):
    model.eval()
    inputs = tokenizer(
        text=text,
        add_special_tokens=True,
        truncation=True,
        return_tensors='pt'
    )
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
    # (1, n, num_labels)
    output = model.forward(inputs['input_ids'], inputs['attention_mask'])
    label_ids = torch.argmax(output, dim=-1).view(-1) #(n)
    result = convert_label_to_entities(tokens, label_ids, label_dict)
    return result

def convert_label_to_entities(tokens: list[str], label_ids: list|torch.Tensor, label_dict:dict):
    id_to_label = [''] * len(label_dict)
    for k in label_dict:
        id_to_label[label_dict[k]] = k
    
    idx = 0
    label_tokens = {}
    combined_tokens = []
    for id in label_ids.tolist():
        label = id_to_label[id]
        token = tokens[idx]
        combined_tokens.append(token + label)
        if label is None or label == 'O':
            idx += 1
            continue
        
        idx += 1
        if label.startswith('B-') or label.startswith('I-'):
            label_tokens[token] = label
    
    return {
        "ner_label": label_tokens,
        "combined_tokens": combined_tokens,
    }

if __name__ == '__main__':
    with open('./config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), yaml.FullLoader)
    
    device = torch.device('cuda' if config['gpu'] and torch.cuda.is_available() else 'cpu')
    label_dict = get_label_dict(os.path.join(config['data_dir'], 'labels.json'))

    tokenizer = BertTokenizerFast.from_pretrained(config['save_model_dir'])
    model = NERModel(config['save_model_dir'], num_labels=len(label_dict), device=device, config=config)
    model.load(config['save_model_dir'])

    text = '时间：2020 年 08 月 06 日 13 时 50 分至 14 时 40 分​ 地点：哈尔滨市南岗区荣市派出所​ 被询问人：林悦晴 性别：女 年龄：28 出生日期：1992 年 8 月 12 日​ 身份证号码：230103199208129876 工作单位及职务：图书馆管理员 联系方式：18845012345​ 讯问内容：在图书馆值班时，一名男子大声接听电话，我上前提醒保持安静，他不仅不听，还故意提高音量，我再次制止，他将咖啡泼在书架上，我阻拦时被他推搡，其他读者报警。'

    result = predict(model, text, tokenizer, label_dict)
    print(result['combined_tokens'])