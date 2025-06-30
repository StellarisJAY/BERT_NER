from transformers import BertModel, BertConfig
from torch import nn
import torch
import os

class NERModel(nn.Module):
    def __init__(self, pretrained_model_dir, num_labels, device, config):
        super().__init__()
        bert_config = BertConfig.from_pretrained(pretrained_model_dir, num_hidden_layers=config['num_hidden_layers'])
        self.bert_config = bert_config
        # BERT Transformer Encoder Layers
        self.bert = BertModel.from_pretrained(pretrained_model_dir, config=bert_config)
        self.bert.to(device)
        # BERT之后的线性神经网络，用来把编码器状态转换成分类输出
        self.dropout = nn.Dropout(config['dropout'])
        self.device = device
        self.num_labels = num_labels
        # 增加分类器部分的正则化
        self.classifier = nn.Sequential(
            nn.Linear(self.bert_config.hidden_size, 256),
            nn.Dropout(config['dropout']),
            nn.ReLU(),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # bert输入文本vocab ids
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        # 分类器输入BERT最后一层编码器的隐藏状态
        out = self.dropout(bert_output.last_hidden_state)
        # 输出 (B, n, num_labels)
        return self.classifier(out)
    
    def load(self, path: str):
        model_file = os.path.join(path, 'pytorch_model.bin')
        self.load_state_dict(torch.load(model_file, map_location=self.device))
    
    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        model_file = os.path.join(path, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_file)
        self.bert_config.save_pretrained(path)