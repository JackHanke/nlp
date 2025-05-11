import math
import time
import sys
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from utils import read_qa_json
from model import QAModel, train_model, test_model

class QADataset(Dataset):
    def __init__(self, data: list):
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return [self.data[idx][0]], torch.Tensor(self.data[idx][1])

def main(verbose: bool = False):  
    torch.manual_seed(0)
    batch_size = 2

    # data preparation    
    train = read_qa_json(file_name='train_complete.jsonl', verbose=verbose)
    valid = read_qa_json(file_name='dev_complete.jsonl')
    test = read_qa_json(file_name='test_complete.jsonl')

    train_ds = QADataset(train)
    val_ds = QADataset(valid)
    test_ds = QADataset(test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Add code to fine-tune and test your MCQA classifier.

    model = QAModel()
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # baseline accuracy
    acc = test_model(model=model, test=test)
    print(f'Baseline test accuracy: {acc*100:.4f}')

    # train model
    train_model(
        model=model, 
        train=train, 
        epochs=1,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    # get final accuracy
    acc = test_model(model=model, test=test)
    print(f'Final test accuracy: {acc*100:.4f}')

    model.save(f'saves/')
           
                 
if __name__ == "__main__":
    # main(verbose=True)
    main(verbose=False)
