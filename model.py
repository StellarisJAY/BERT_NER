from transformers import BertModel, BertConfig
from torch import nn
import torch

class NERModel(nn.Module):
    def __init__(self, pretrained_model_dir, num_labels, device):
        super().__init__()
        bert_config = BertConfig.from_pretrained(pretrained_model_dir)
        # BERT Transformer Encoder Layers
        self.bert = BertModel.from_pretrained(pretrained_model_dir, config=bert_config)
        self.bert.to(device)
        # BERT之后的线性神经网络，用来把编码器状态转换成分类输出
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels, device=device)
        self.dropout = nn.Dropout(0.1)
        self.device = device

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # bert输入文本vocab ids
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        # 分类器输入BERT最后一层编码器的隐藏状态
        out = self.dropout(bert_output.last_hidden_state)
        # 输出 (B, n, num_labels)
        return self.classifier(out)