import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

from main import OpenBookQADataset, custom_collate, evaluate

BERT_MODEL = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 4
EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
PREFIX_LEN = 5

class PrefixEncoder(nn.Module):
    def __init__(self, prefix_length, hidden_size):
        super().__init__()
        self.prefix = nn.Parameter(torch.randn(prefix_length, hidden_size))

    def forward(self, batch_size):
        return self.prefix.unsqueeze(0).expand(batch_size, -1, -1)

# train
def train_prefix():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    # Add the special prefix token
    prefix_token = '[QAPrefix]'
    tokenizer.add_special_tokens({'additional_special_tokens': [prefix_token]})

    # Pass prefix token to dataset
    train_set = OpenBookQADataset("train_complete.jsonl", tokenizer, prefix=prefix_token)
    val_set = OpenBookQADataset("dev_complete.jsonl", tokenizer, prefix=prefix_token)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=custom_collate)

    model = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True).to(DEVICE)
    # freeze all weights 
    for param in model.parameters():
        param.requires_grad = False
    
    # resize for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    hidden_size = model.config.hidden_size
    classifier = nn.Linear(hidden_size, 4).to(DEVICE) 

    optimizer = torch.optim.AdamW(classifier.parameters(), lr=1e-4)

    learnable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {learnable_params:,}')

    print("Starting Prefix Tuning...")
    start = time.time()
    model.train()
    classifier.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for encodings, labels in train_loader:
            labels = torch.tensor(labels).to(DEVICE)
            optimizer.zero_grad()

            outputs = model(encodings)
            logits = outputs.last_hidden_state[:, 0, :]  # For CLS-based classification

            loss = nn.CrossEntropyLoss()(classifier(logits), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.2f}")
    train_time = time.time() - start

    # Evaluate with classifier (no prefix_encoder)
    val_acc, val_infer_time = evaluate(model, val_loader, classifier=classifier)
    print(f"Validation Accuracy: {val_acc:.3f} | Inference Time: {val_infer_time:.2f}s")
    print(f"Training Time: {train_time:.2f}s")

    return model, classifier


# main
if __name__ == "__main__":
    trained_model, trained_classifier = train_prefix()
