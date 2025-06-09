import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from torch.optim import AdamW


BERT_MODEL = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_collate(batch):
    encodings = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return encodings, labels

# dataset
class OpenBookQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, prefix: str = None):
        self.data = []
        with open(jsonl_path, "r") as f:
            for line in f:
                ex = json.loads(line)
                inputs = []
                fact = ex.get("fact1", "")
                stem = ex["question"]["stem"]
                choices = ex["question"]["choices"]
                label = next(i for i, c in enumerate(choices) if c["label"] == ex["answerKey"])
                for choice in choices:
                    if prefix is None:
                        text = f"{fact} {stem} {choice['text']}"
                    else:
                        text = f"{prefix}{fact} {stem} {choice['text']}"

                    enc = tokenizer(text, truncation=True, padding="max_length", max_length=MAX_LENGTH, return_tensors="pt")
                    # print(type(enc['input_ids']))
                    # print(type(enc))

                    # if prefix is not None:
                    #     # prune automated [CLS] token added by tokenizer
                    #     enc['input_ids'] = torch.cat((enc['input_ids'][:, 1:], torch.tensor([[0]])), dim=1)
                    # print(type(enc))

                    inputs.append(enc)
                self.data.append((inputs, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encodings, label = self.data[idx]
        return encodings, label

# model
class CustomBertMC(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, batch_encodings):
        logits = []
        for i in range(4):  

            input_dict = {
                k: torch.cat([ex[i][k] for ex in batch_encodings], dim=0).to(DEVICE)
                for k in batch_encodings[0][i].keys()
            }
            out = self.bert(**input_dict)
            cls_embed = out.last_hidden_state[:, 0, :]  # [CLS] embedding
            logit = self.classifier(cls_embed) 
            logits.append(logit)

        logits = torch.cat(logits, dim=1) 
        return logits


# training
def evaluate(model, dataloader):
    model.eval()
    correct = total = 0
    start_time = time.time()
    with torch.no_grad():
        for encodings, labels in dataloader:
            logits = model(encodings)
            preds = torch.argmax(logits, dim=1).cpu()
            correct += (preds == torch.tensor(labels)).sum().item()
            total += len(labels)
    acc = correct / total
    elapsed = time.time() - start_time
    return acc, elapsed

def train():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    train_set = OpenBookQADataset("train_complete.jsonl", tokenizer)
    val_set = OpenBookQADataset("dev_complete.jsonl", tokenizer)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=custom_collate)


    model = CustomBertMC().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {learnable_params:,}')

    print("Starting training...")
    train_start = time.time()
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
    train_time = time.time() - train_start

    val_acc, val_infer_time = evaluate(model, val_loader)
    print(f"Validation Accuracy: {val_acc:.3f} | Inference Time: {val_infer_time:.2f}s")
    print(f"Training Time: {train_time:.2f}s")

    return model

# main
if __name__ == "__main__":
    trained_model = train()
