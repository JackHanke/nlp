import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from main import evaluate

BERT_MODEL = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
PREFIX_LEN = 5


class PrefixTuningModel(nn.Module):
    def __init__(self, base_model, prefix_length, num_labels):
        super().__init__()
        self.base_model = base_model
        self.prefix = nn.Parameter(torch.randn(prefix_length, base_model.config.hidden_size))
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None):
        # get input embeddings from base model
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        batch_size = input_embeds.size(0)
        # expand and concatenate prefix embeddings
        prefix_embeds = self.prefix.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        # Extend attention mask if provided
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, self.prefix.size(0), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        # Forward through base model
        outputs = self.base_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        cls_hidden = outputs.last_hidden_state[:, self.prefix.size(0), :]
        logits = self.classifier(cls_hidden)
        return logits


# Local only
def collate_fn(batch):
    input_ids = torch.stack([item[0]['input_ids'] for item in batch])
    attention_mask = torch.stack([item[0]['attention_mask'] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}, labels


# for the local
class PrefixOpenBookQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer):
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
                    text = f"{fact} {stem} {choice['text']}"
                    enc = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
                    inputs.append(enc)
                self.data.append((inputs, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        encodings, label = self.data[idx]
        selected_encoding = {
            'input_ids': encodings[label]['input_ids'].squeeze(0),
            'attention_mask': encodings[label]['attention_mask'].squeeze(0)
        }
        return selected_encoding, label
    

# train
def train_prefix():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    train_set = PrefixOpenBookQADataset("train_complete.jsonl", tokenizer)
    val_set = PrefixOpenBookQADataset("dev_complete.jsonl", tokenizer)
    num_labels = len(train_set.data[0][0])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Load and freeze base BERT model
    base_model = BertModel.from_pretrained(BERT_MODEL).to(DEVICE)
    for param in base_model.parameters():
        param.requires_grad = False

    # Wrap in PrefixTuningModel
    model = PrefixTuningModel(base_model, PREFIX_LEN, num_labels).to(DEVICE)

    # Only optimize the prefix parameters
    optimizer = torch.optim.AdamW([model.prefix], lr=1e-4)

    print("Starting Prefix Tuning...")
    start = time.time()
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for encodings, labels in train_loader:
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            input_ids = encodings['input_ids'].to(DEVICE)
            attention_mask = encodings['attention_mask'].to(DEVICE)
            # Forward pass through prefix-tuning wrapper
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.2f}")
    train_time = time.time() - start

    # Evaluate without classifier
    val_acc, val_infer_time = evaluate(model, val_loader)
    print(f"Validation Accuracy: {val_acc:.3f} | Inference Time: {val_infer_time:.2f}s")
    print(f"Training Time: {train_time:.2f}s")

    return model, None


# main
if __name__ == "__main__":
    trained_model, trained_prefix_embeddings = train_prefix()
