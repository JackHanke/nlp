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

    train_set = OpenBookQADataset("train_complete.jsonl", tokenizer)
    val_set = OpenBookQADataset("dev_complete.jsonl", tokenizer)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=custom_collate)

    model = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True).to(DEVICE)
    # freeze all weights 
    for param in model.parameters():
        param.requires_grad = False
    
    # resize for the new tokens
    model.resize_token_embeddings(len(tokenizer))

    hidden_size = model.config.hidden_size
    prefix_encoder = PrefixEncoder(PREFIX_LEN, hidden_size).to(DEVICE)
    classifier = nn.Linear(hidden_size, 4).to(DEVICE) 

    optimizer = torch.optim.AdamW(prefix_encoder.parameters(), lr=1e-4)  

    learnable_params = sum(p.numel() for p in prefix_encoder.parameters() if p.requires_grad) + sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f'Number of learnable parameters: {learnable_params:,}')

    print("Starting Prefix Tuning...")
    start = time.time()
    model.train()
    prefix_encoder.train()
    classifier.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for encodings, labels in train_loader:
            labels = torch.tensor(labels).to(DEVICE)
            optimizer.zero_grad()
            input_ids = encodings['input_ids'].to(DEVICE)
            attention_mask = encodings['attention_mask'].to(DEVICE)
            batch_size = input_ids.size(0)

            inputs_embeds = model.embeddings(input_ids)
            prefix_embeds = prefix_encoder(batch_size)
            inputs_embeds = torch.cat((prefix_embeds, inputs_embeds), dim=1)

            extended_attention_mask = torch.cat((torch.ones(batch_size, PREFIX_LEN).to(DEVICE), attention_mask), dim=1)

            outputs = model(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)
            logits = outputs.last_hidden_state[:, 0, :]  # For CLS-based classification

            loss = nn.CrossEntropyLoss()(classifier(logits), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.2f}")
    train_time = time.time() - start

    # Evaluate with prefix_encoder and classifier
    val_acc, val_infer_time = evaluate(model, val_loader, prefix_encoder=prefix_encoder, classifier=classifier)
    print(f"Validation Accuracy: {val_acc:.3f} | Inference Time: {val_infer_time:.2f}s")
    print(f"Training Time: {train_time:.2f}s")

    return model, prefix_encoder, classifier


# main
if __name__ == "__main__":
    trained_model, trained_prefix_encoder, trained_classifier = train_prefix()
