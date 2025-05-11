import torch
import torch.nn as nn
from transformers import AutoTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer

# 
class QAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")
        self.linear = torch.rand(768, 1)

    def forward(self, sentences: list[str]):
        for question in questions:
        outputs = []
        for sentence in sentences:
            tokens = self.tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            x = self.bert_model(**tokens).last_hidden_state[0][0].unsqueeze(0) # get [CLS] token
            outputs.append(x)
        output_batch = torch.cat(outputs)
        x = output_batch @ self.linear # enbedding for [CLS] token
        x = torch.transpose(x, 1, 0)
        return x
