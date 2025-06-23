import json
from torch.utils.data import Dataset, DataLoader
import torch

def load_data(data_dir):
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
    def __init__(self, file_path, label_dict, tokenizer, max_len, samples=1000):
        super().__init__()
        self.samples = samples
        self.label_dict = label_dict
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.loads(f.read())
        self.samples = min(len(self.data), samples)
        self.texts = [item["text"] for item in self.data][:self.samples]
        self.labels = [item["label"] for item in self.data][:self.samples]
    
    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # tokenizer + [CLS][SEP] + to_vocab_id
        inputs = self.tokenizer.encode_plus(
            self.texts[idx],
            None,
            add_special_tokens=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        label_ids = [self.label_dict[label] for label in self.labels[idx]]
        # [CLS]和[SEP]token标记为O
        label_ids.insert(0, 0)
        label_ids.append(0)

        if len(label_ids) > self.max_len:
            label_ids = label_ids[:self.max_len]
        if len(label_ids) < self.max_len:
            label_ids.extend([0] * (self.max_len - len(label_ids)))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
        }