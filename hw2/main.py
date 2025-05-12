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

# NOTE Question 1: fine tune BERT
def bert_finetuning(
        train: list,
        valid: list,
        test: list,

    ):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = QAModel(device=device)
    optimizer = optim.Adam(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    # baseline accuracy
    acc = test_model(model=model, test=test)
    print(f'Baseline test accuracy: {acc*100:.4f}')

    # train model
    train_model(
        model=model, 
        train=train, 
        valid=valid,
        epochs=1,
        optimizer=optimizer,
        loss_fn=loss_fn,
    )

    # get final accuracy
    acc = test_model(model=model, test=test)
    print(f'Final test accuracy: {acc*100:.4f}')

    # save model
    model.save()

# NOTE Question 2: 



def main(verbose: bool = False):  
    torch.manual_seed(0)
    batch_size = 2

    # data preparation    
    train = read_qa_json(file_name='train_complete.jsonl', verbose=verbose)
    valid = read_qa_json(file_name='dev_complete.jsonl')
    test = read_qa_json(file_name='test_complete.jsonl')

    # Add code to fine-tune and test your MCQA classifier.
    
    # NOTE Question 1 
    # bert_finetuning(
    #     train=train,
    #     valid=valid,
    #     test=test,
    # )
    

    
           
                 
if __name__ == "__main__":
    # main(verbose=True)
    main(verbose=False)
