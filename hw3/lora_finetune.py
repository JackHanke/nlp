# system imports
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
# local imports
from main import OpenBookQADataset, custom_collate, evaluate

BERT_MODEL = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

# LoRA
class LoRAdapter(nn.Module):
    # using notation from original paper
    def __init__(self, hidden_size, bottleneck_size=6, dropout=0.1):
        super().__init__()
        # no biases
        self.A = nn.Linear(hidden_size, bottleneck_size, bias=False)
        self.B = nn.Linear(bottleneck_size, hidden_size, bias=False)
        # zero weights of B in
        nn.init.zeros_(self.B.weight)
        nn.init.zeros_(self.B.bias)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.A(x)
        x = self.B(x)
        return self.dropout(x)

class BertWithLoRA(nn.Module):
    def __init__(self, adapter_dim=128):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True)
        self.adapters = nn.ModuleList([
            LoRAdapter(self.bert.config.hidden_size, adapter_dim)
            for _ in range(self.bert.config.num_hidden_layers)
        ])
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

        # freeze BERT params
        for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, batch_encodings):
        logits = []
        for i in range(4):  # 4 as expected
            input_dict = {
                k: torch.cat([ex[i][k] for ex in batch_encodings], dim=0).to(DEVICE)
                for k in batch_encodings[0][i].keys()
            }
            output = self.bert(**input_dict)
            hidden_states = list(output.hidden_states)

            for i, adapter in enumerate(self.adapters):
                hidden_states[i + 1] = adapter(hidden_states[i + 1])
            cls_embed = hidden_states[-1][:, 0, :]
            
            # now only adapting the last hidden layer
            #adapted = self.adapters[-1](hidden_states[-1])
            #cls_embed = adapted[:, 0, :]

            logit = self.classifier(cls_embed)
            logits.append(logit)

        logits = torch.cat(logits, dim=1)
        return logits


# train
def train_lora():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train_set = OpenBookQADataset("train_complete.jsonl", tokenizer)
    val_set = OpenBookQADataset("dev_complete.jsonl", tokenizer)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=custom_collate)

    model = BertWithLoRA().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  

    print("Starting Adapter Tuning...")
    start = time.time()
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for encodings, labels in train_loader:
            labels = torch.tensor(labels).to(DEVICE)
            optimizer.zero_grad()
            logits = model(encodings)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.2f}")
    train_time = time.time() - start

    val_acc, val_infer_time = evaluate(model, val_loader)
    print(f"Validation Accuracy: {val_acc:.3f} | Inference Time: {val_infer_time:.2f}s")
    print(f"Training Time: {train_time:.2f}s")

    return model


# main
if __name__ == "__main__":
    trained_adapter_model = train_lora()
