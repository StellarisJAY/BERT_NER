from transformers import BertModel, BertTokenizer
from torch import nn
import torch
from dataset import NERDataset, load_data
import json
from torch.utils.data import DataLoader
from torch.nn import functional as F
from model import NERModel
from tqdm import tqdm

PRETRAINED_MODEL_DIR = "D:\\models\\chinese-roberta-wwm-ext"

def train_ner(train_loader: DataLoader, valid_loader: DataLoader, model: BertModel, num_epochs, lr=0.003):
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = model.device
    print('=============开始训练==============')
    for epoch in range(num_epochs):
        train_losses = []
        for inputs in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            label_ids = F.one_hot(inputs['label_ids'].to(device)).float()
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            l = loss(output, label_ids).mean()
            l.backward()
            optimizer.step()
            train_losses.append(l)
        
        with torch.no_grad():
            print(f'epoch:{epoch},train_loss={torch.tensor(train_losses).mean()}')
            valid_losses = []
            for inputs in valid_loader:
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                label_ids = F.one_hot(inputs['label_ids'].to(device)).float()
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                l = loss(output, label_ids).mean()
                valid_losses.append(l)
            print(f'epoch:{epoch},valid_loss={torch.tensor(valid_losses).mean()}')


def get_label_dict(file:str):
    with open(file, 'r', encoding='utf-8') as f:
        return json.loads(f.read())

if __name__ == "__main__":
    batch_size = 32

    tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_DIR)
    device = torch.device("cpu")

    load_data('data')
    label_dict = get_label_dict('data/labels.json')
    # 训练数据集
    train_dataset = NERDataset('data/train_label.json', label_dict, tokenizer, max_len=128, samples=512)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    # 验证数据集
    valid_dataset = NERDataset('data/valid_label.json', label_dict, tokenizer, max_len=128, samples=64)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size)

    model = NERModel(PRETRAINED_MODEL_DIR, num_labels=len(label_dict), device=device)

    train_ner(train_loader, valid_loader, model, num_epochs=2, lr=0.003)


        