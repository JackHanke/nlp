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

# train
def train_prefix():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    prefix_token = '[QAPrefix]'
    special_tokens_dict = {'additional_special_tokens': [prefix_token]}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # make prefix dataset
    train_set = OpenBookQADataset("train_complete.jsonl", tokenizer, prefix=prefix_token)
    val_set = OpenBookQADataset("dev_complete.jsonl", tokenizer, prefix=prefix_token)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, collate_fn=custom_collate)

    model = BertModel.from_pretrained(BERT_MODEL, output_hidden_states=True).to(DEVICE)
    # freeze all weights 
    for param in model.parameters():
        param.requires_grad = False
    
    # add extra embedding to BERT for prefix
    model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  

    print("Starting Prefix Tuning...")
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
    trained_adapter_model = train_prefix()
