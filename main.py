from transformers import BertTokenizerFast
from torch import nn
import torch
from dataset import NERDataset, load_data
import json
from torch.utils.data import DataLoader
from model import NERModel
from tqdm import tqdm
import yaml

def train_ner(train_loader: DataLoader, valid_loader: DataLoader, model: NERModel, num_epochs=5, lr=3e-5):
    model.train(True)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = model.device
    print('=============开始训练==============')
    for epoch in range(num_epochs):
        train_losses = []
        train_acc = []
        for inputs in tqdm(train_loader):
            optimizer.zero_grad()
            input_ids = inputs['input_ids'].to(device) # (B,n)
            attention_mask = inputs['attention_mask'].to(device) #(B,n)
            label_ids = inputs['label_ids'].to(device) # (B,n)
            # forward
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            l = loss(output.view(-1, model.num_labels), label_ids.view(-1)) # (B*n,num_labels), (B*n)
            l.backward()
            optimizer.step()
            train_losses.append(l)
            train_acc.append(accuracy(output, inputs['label_ids']))
        
        with torch.no_grad():
            print(f'epoch:{epoch},train_loss={torch.tensor(train_losses).mean()},train_acc={torch.tensor(train_acc).mean()}')
        (valid_loss,valid_acc) = valid(valid_loader, model, loss)
        print(f'epoch:{epoch},valid_loss={valid_loss},valid_acc={valid_acc}')
    model.train(False)

def accuracy(pred: torch.Tensor, actual: torch.Tensor):
    pred = pred.argmax(dim=-1)
    return (pred == actual).float().mean()

def valid(valid_loader: DataLoader, model: NERModel, loss):
    with torch.no_grad():
        valid_losses = []
        valid_acc = []
        for inputs in valid_loader:
            input_ids = inputs['input_ids'].to(device) # (B, n)
            attention_mask = inputs['attention_mask'].to(device) # (B, n)
            label_ids = inputs['label_ids'].to(device) # (B, n)
            # forward
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            l = loss(output.view(-1, model.num_labels), label_ids.view(-1))
            valid_losses.append(l)
            valid_acc.append(accuracy(output, inputs['label_ids']))
        return (torch.tensor(valid_losses).mean().item(), torch.tensor(valid_acc).mean().item())

def get_label_dict(file:str):
    with open(file, 'r', encoding='utf-8') as f:
        return json.loads(f.read())
    
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
    print(result)
    pass

def convert_label_to_entities(tokens: list[str], label_ids: list|torch.Tensor, label_dict:dict):
    id_to_label = [''] * len(label_dict)
    for k in label_dict:
        id_to_label[label_dict[k]] = k
    
    idx = 0
    label_tokens = {}
    for id in label_ids.tolist():
        label = id_to_label[id]
        if label is None or label == 'O':
            idx += 1
            continue
        token = tokens[idx]
        idx += 1
        if label.startswith('B-') or label.startswith('I-'):
            label_tokens[token] = label
    
    return {
        "ner_label": label_tokens
    }



if __name__ == "__main__":
    with open('config.yaml') as f:
        config = yaml.load(f.read(), yaml.FullLoader)
    
    print(f'config: {config}')
    batch_size = config['batch_size']
    max_length = config['max_length']
    train_samples = config['train_samples']
    valid_samples = config['valid_samples']

    num_epochs = config['num_epochs']
    lr = config['lr']

    device = torch.device('cuda') if config['gpu'] and torch.cuda.is_available() else torch.device('cpu')
    
    # BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config['pretrained_model'])
   

    # 加载训练+验证数据
    if config['train']:
        load_data(data_dir='data')
    # label 标记
    label_dict = get_label_dict(file=config['data_dir'] + '/labels.json')
    # 加载模型
    model = NERModel(config['pretrained_model'], num_labels=len(label_dict), device=device)

    if config['train']:
        # 训练数据集
        train_dataset = NERDataset(config['data_dir'] + '/train_label.json', label_dict, tokenizer, max_len=max_length, samples=train_samples)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        # 验证数据集
        valid_dataset = NERDataset(config['data_dir'] + '/valid_label.json', label_dict, tokenizer, max_len=max_length, samples=valid_samples)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=8)
        # 训练
        train_ner(train_loader, valid_loader, model, num_epochs=num_epochs, lr=lr)

        