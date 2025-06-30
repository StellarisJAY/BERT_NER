from transformers import BertTokenizerFast
from torch import nn
import torch
from dataset import NERDataset, get_label_dict
from torch.utils.data import DataLoader
from model import NERModel
from tqdm import tqdm
import yaml
import os

def train_ner(train_loader: DataLoader, valid_loader: DataLoader, model: NERModel, num_epochs=5, lr=3e-5):
    model.train(True)
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    device = model.device
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
            # 修改后应增加mask过滤
            # mask = (label_ids.view(-1) != 0).float()  # 假设0是padding的标签
            # l = loss(output.view(-1, model.num_labels), label_ids.view(-1)) * mask
            # l = l.sum() / mask.sum()
            l = loss(output.view(-1, model.num_labels), label_ids.view(-1)).mean()
            l.backward()
            optimizer.step()
            train_losses.append(l)
            train_acc.append(accuracy(output, inputs['label_ids']))
        
        with torch.no_grad():
            print(f'epoch:{epoch},train_loss={torch.tensor(train_losses).mean()},train_acc={torch.tensor(train_acc).mean().item() * 100:.2f}%')
            (valid_loss,valid_acc) = valid(valid_loader, model, loss)
            print(f'epoch:{epoch},valid_loss={valid_loss},valid_acc={valid_acc * 100:.2f}%')
    model.train(False)

def accuracy(pred: torch.Tensor, actual: torch.Tensor):
    pred = pred.argmax(dim=-1)
    return (pred == actual).float().mean()

def valid(valid_loader: DataLoader, model: NERModel, loss):
    valid_losses = []
    valid_acc = []
    for inputs in tqdm(valid_loader):
        input_ids = inputs['input_ids'].to(device) # (B, n)
        attention_mask = inputs['attention_mask'].to(device) # (B, n)
        label_ids = inputs['label_ids'].to(device) # (B, n)
        # forward
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        l = loss(output.view(-1, model.num_labels), label_ids.view(-1)).mean()
        valid_losses.append(l)
        valid_acc.append(accuracy(output, inputs['label_ids']))
    return (torch.tensor(valid_losses).mean().item(), torch.tensor(valid_acc).mean().item())

def test(test_loader: DataLoader, model: NERModel):
    model.eval()
    accs = []
    for inputs in test_loader:
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)
        label_ids = inputs['label_ids'].to(model.device)
        pred = model(input_ids, attention_mask)
        accs.append(accuracy(pred, label_ids))
    print(f'Test accuracy: {torch.tensor(accs).mean().item() * 100:.2f}%')

if __name__ == "__main__":
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), yaml.FullLoader)
    print(f'config: {config}')
    batch_size = config['batch_size']
    max_length = config['max_length']
    num_epochs = config['num_epochs']
    lr = float(config['lr'])

    device = torch.device('cuda') if config['gpu'] and torch.cuda.is_available() else torch.device('cpu')
    # BERT tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(config['pretrained_model'])
    # label 标记
    label_dict = get_label_dict(file=config['data_dir'] + '/labels.json')
    # 加载模型
    model = NERModel(config['pretrained_model'], num_labels=len(label_dict), device=device, config=config)
    print(model)
    print('加载数据集...')
    # 训练数据集
    train_dataset = NERDataset(config['data_dir'] + '/train_label.json', label_dict, tokenizer, max_len=max_length)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    # 验证数据集
    valid_dataset = NERDataset(config['data_dir'] + '/valid_label.json', label_dict, tokenizer, max_len=max_length)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, num_workers=8)
    # 测试数据集
    test_file = os.path.join(config['data_dir'], 'test_label.json')
    test_dataset = None
    if os.path.exists(test_file):
        test_dataset = NERDataset(test_file, label_dict, tokenizer, max_length)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)
    print(f'训练数据:{len(train_dataset)}条，验证集:{len(valid_dataset)}条')
    if test_dataset is not None:
        print(f'测试集:{len(test_dataset)}条')
    print('开始训练...')
    # 训练
    train_ner(train_loader, valid_loader, model, num_epochs=num_epochs, lr=lr)
    # 保存模型参数
    model.save(config['save_model_dir'])
    tokenizer.save_pretrained(config['save_model_dir'])
    print('训练结束，已保存模型')
    if test_dataset:
        # 测试
        with torch.no_grad():
            test(test_loader, model)
    

        