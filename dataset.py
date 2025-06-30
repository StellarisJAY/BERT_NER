import json
from torch.utils.data import Dataset
import torch
import os

def get_label_dict(file:str):
    with open(file, 'r', encoding='utf-8') as f:
        return json.loads(f.read())

def load_data(data_dir):
    train_file, valid_file, labels_file = data_dir+'/train_label.json', data_dir+'/valid_label.json', data_dir+'/labels.json'
    if os.path.exists(train_file) and os.path.exists(valid_file) and os.path.exists(labels_file):
        return

    with open(data_dir + "/train.json", 'r', encoding='utf-8') as train:
        train_lines = train.readlines()
    with open(data_dir + "/dev.json", 'r', encoding='utf-8') as dev:
        dev_lines = dev.readlines()

    label_dict = {'O': 0}
    train_data = convert_data(train_lines, label_dict)
    dev_data = convert_data(dev_lines, label_dict)
    
    with open(data_dir + '/train_label.json', 'w+', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(data_dir + '/valid_label.json', 'w+', encoding='utf-8') as f:
        json.dump(dev_data, f, ensure_ascii=False)
    with open(data_dir + '/labels.json', 'w+', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False)

def convert_data(lines: list[str], label_dict: dict):
    # {"text": "浙商银行企业信贷部叶老桂博士则从另一个角度对五道门槛进行了解读。叶老桂认为，对目前国内商业银行而言，", 
        # "label": {"name": {"叶老桂": [[9, 11]]}, "company": {"浙商银行": [[0, 3]]}}}
    # 转换成B-xx I-xx O 标签格式
    data = []
    for line in lines:
        row = json.loads(line)
        labels = ["O"] * len(row['text'])
        for key in row["label"]:
            label = key.upper()
            b_label, i_label = 'B-'+label, 'I-'+label
            if label_dict.get(b_label) is None:
                label_dict[b_label] = len(label_dict)
            if label_dict.get(i_label) is None:
                label_dict[i_label] = len(label_dict)
            for positions in row['label'][key].values():
                for pos in positions:
                    labels[pos[0]] = b_label
                    labels[pos[0]+1:pos[1]+1] = [i_label] * (pos[1]-pos[0])
        row['label'] = labels
        data.append(row)
    return data


class NERDataset(Dataset):
    def __init__(self, file_path, label_dict, tokenizer, max_len):
        super().__init__()
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.loads(f.read())
        self.samples = len(self.data)
        self.texts = [item["text"] for item in self.data][:self.samples]
        self.labels = [item["label"] for item in self.data][:self.samples]

        self.id_to_label = [''] * len(self.label_dict)
        for k in self.label_dict:
            self.id_to_label[self.label_dict[k]] = k
        self.processed_data = []
        for i in range(self.samples):
            self.processed_data.append(self.process_data(self.texts[i], self.labels[i]))

    def process_data(self, text: str, labels: list):
        # tokenize + [CLS][SEP] + to_vocab_id
        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            padding='max_length',
            max_length=self.max_len,
            truncation='longest_first',
            return_offsets_mapping=True
        )
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        label_ids = self.align_label(self.label_dict, inputs['offset_mapping'], labels)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
        }
    
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        return self.processed_data[idx]
        
    def align_label(self, label_dict: dict, offset_mapping: list[tuple], labels: list[str]):
        label_ids = [0] * len(offset_mapping)
        for i in range(len(offset_mapping)):
            (start, end) = offset_mapping[i]
            if start == 0 and end == 0:
                continue
            label_ids[i] = label_dict[labels[start]]
        return label_ids
