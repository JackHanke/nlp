import argparse
import os
import sys
import shutil
import random
import numpy as np
import time
import copy
import math
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torcheval.metrics.text import Perplexity
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader, random_split

from torch.amp import autocast, GradScaler

from model import Transformer

# read corpus and structure as tensor
def read_corpus(filename: str, tokenizer: callable):
    seq = []
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip() 
            if line: 
                tokens = tokenizer(line)
                seq.extend(tokens['input_ids'])
    return torch.tensor(seq, dtype=torch.long).contiguous()

# sequentially loop over batch 
def get_batches(data: torch.Tensor, seq_len: int, batch_size: int, device: torch.device, offset: int):
    # get adjusted data size
    n_batches = (data.size(0)-offset) // (seq_len * batch_size)
    # offset data to avoid bias in context, token pairing
    data = data[offset: (n_batches * batch_size * seq_len)+offset]
    # structure as batch
    data = data.view(batch_size, -1)
    # create generator for batched data
    for i in range(0, data.size(1) - seq_len, seq_len):
        x = data[:, i:i+seq_len]
        y = data[:, i+1:i+1+seq_len]
        yield x.to(device), y.to(device)

# create model
def get_model(opt, src_vocab, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout, opt.seqlen, opt.device)
    model.to(opt.device)
       
    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model

# training script     
def train_model(model: torch.nn.Module, opt: any):
    # set model to train mode
    model.train()

    # logging values
    training_perplexities, valid_perplexities = [], []
    best_valid_perplexity = float('inf')

    # make scaler for casting
    scaler = GradScaler()

    # training loop
    for epoch in range(opt.epochs):
        # training perplexity
        train_metric = Perplexity()
        train_metric.to(opt.device)

        # Get data batches
        data_iter = get_batches(data=opt.train, seq_len=opt.seqlen, batch_size=opt.batchsize, device=opt.device, offset=epoch)
        prog_bar = tqdm(enumerate(data_iter))
        

        for i, (x_batch, y_batch) in prog_bar:
            # autocast for lower precision training
            with autocast(device_type=y_batch.device.type, dtype=torch.float16):
                # zero gradients
                opt.optimizer.zero_grad()

                # inference
                predictions = model.forward(trg=x_batch)
                # print(opt.vocab_size)
                # print(y_batch.shape)
                # input(predictions.shape)
                # print(y_batch.view(-1))

                y_batch = y_batch.contiguous()
                # input(predictions.view(-1, opt.vocab_size))

                #  4. linearize the predictions and compute the loss against ground truth
                # loss
                train_loss_val = F.cross_entropy(predictions.view(-1, opt.vocab_size), y_batch.view(-1), ignore_index=-1)

                prog_bar.set_description(f'Loss: {train_loss_val:.6f}')
                # update perplexity metric
                train_metric.update(predictions, y_batch)

            #  5. calculate and apply the gradients with loss.backward() and optimizer.step()
            scaler.scale(train_loss_val).backward()
            scaler.step(opt.optimizer)
            scaler.update()

        #  6. report intermediate trainining perplexity
        training_perplexity = train_metric.compute()
        val = training_perplexity.cpu().item()
        update_string = f'Train epoch: {epoch} Perplexity: {val}'
        print(update_string)
        prog_bar.set_description(update_string)
        training_perplexities.append(val)

        #  7. generate a test perplexity once per training epoch by calling test_model()
        valid_perplexity = test_model(model=model, opt=opt, epoch=epoch)
        val = valid_perplexity.cpu().item()
        update_string = f'Val epoch: {epoch} Perplexity: {val}'
        print(update_string)
        prog_bar.set_description(update_string)
        valid_perplexities.append(val)

        #  8. save model weights to file specified in opt.savename
        if valid_perplexity < best_valid_perplexity:
            torch.save(model.state_dict(), opt.savename)
            best_valid_perplexity = valid_perplexity
    
    return training_perplexities, valid_perplexities


@torch.no_grad()
# test model
def test_model(model: torch.nn.Module, opt: any, epoch: int):
    # write code to generate perplexity of test set
    
    model.eval()
    
    metric = Perplexity()
    metric.to(opt.device)

    if epoch >= 0:
        data_iter = get_batches(data=opt.valid, seq_len=opt.seqlen, batch_size=opt.batchsize, device=opt.device, offset=epoch)
    # NOTE the starter code calls testing "epoch -1"
    elif epoch < 0:
        data_iter = get_batches(data=opt.test, seq_len=opt.seqlen, batch_size=opt.batchsize, device=opt.device, offset=epoch)

    for i, (x_batch, y_batch) in enumerate(data_iter):

        # inference
        predictions = model.forward(
            trg=x_batch
        )

        # update perplexity metric
        metric.update(predictions, y_batch)
    

    perplexity_value = metric.compute()
    
    model.train()
    return perplexity_value

# 
def main():
    random.seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true') # NOTE store true means "default false" ???
    parser.add_argument('-SGDR', action='store_true')
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-batchsize', type=int, default=1)
    parser.add_argument('-printevery', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.00001)
    parser.add_argument('-seqlen', type=int, default=512)
    parser.add_argument('-threshold', type=int, default=3)
    parser.add_argument('-savename', type=str)    
    parser.add_argument('-loadname', type=str)    
    parser.add_argument('-tied', type=int, default=1)
    parser.add_argument('-dir_name', type=str,default='model')
    parser.add_argument('-norm', type=float, default=2.0)
    parser.add_argument('-dataset', type=str, default='wiki2')
                
    opt = parser.parse_args()
    opt.verbose = False    
    
    # NOTE this is changed from starter because it required cuda device
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    
    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    dir_name = "saved/%s" % (opt.dir_name)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    source_name = sys.argv[0]
    dir_name = dir_name + "//"
    opt.dir_name = dir_name
    shutil.copy(source_name,dir_name + source_name)
    opt.log_file = dir_name + "log_file.txt"
    
    print(str(opt))
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    if opt.dataset == 'wiki2':
        opt.train = read_corpus('data/wiki2.train.txt',tokenizer)
        opt.valid = read_corpus('data/wiki2.valid.txt',tokenizer)
        opt.test = read_corpus('data/wiki2.test.txt',tokenizer)
    elif opt.dataset == 'wiki103':
        opt.train = read_corpus('data/wiki103.train.txt',tokenizer)
        opt.valid = read_corpus('data/wiki103.valid.txt',tokenizer)
        opt.test = read_corpus('data/wiki103.test.txt',tokenizer)
    
    obs = len(opt.train)
    opt.vocab_size = 50257
    temp = []
    for i in range(opt.vocab_size):
        temp.append(i)
    opt.indices = torch.tensor(temp)
    if not opt.no_cuda: opt.indices = opt.indices.cuda()
    
    model = get_model(opt,opt.vocab_size,opt.vocab_size)
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    text = 'total params: %d' % (params)
    print(text)

    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    if opt.SGDR == True:
        opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)

    opt.src_pad = 0
    opt.trg_pad = 0
            
    train_perplexities, valid_perplexities = train_model(model, opt)
    test_perplexity = test_model(model, opt, -1)

    # learning curve plotting
    plt.plot([i+1 for i in range(len(train_perplexities))],[val for val in train_perplexities],label=f'Train Perplexity')
    plt.plot([i+1 for i in range(len(valid_perplexities))],[val for val in valid_perplexities],label=f'Validation Perplexity')
    plt.legend()
    plt.ylabel(f'Perplexity')
    plt.xlabel(f'Epochs')
    plt.title(f'{opt.dataset} Training/Validation Curves')
    # plt.show()
    plt.savefig(f'./wiki2_learning_curve.png')

    print(f'Test perplexity: {test_perplexity}')
        
if __name__ == "__main__":
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    main()

    # NOTE d_model is the embedding dimension

    '''
    debug command for making tiny model, good for local dev: 
    
    python3 starter.py \
        -dir_name model_wiki2 \
        -savename "saved/model_wiki2/model.pth" \
        -batchsize 2 \
        -n_layers 1 \
        -d_model 16 \
        -epochs 2 \
        -no_cuda

    full training command for wiki2:

    python3 starter.py \
        -dir_name model_wiki2 \
        -savename "saved/model_wiki2/model.pth" \
        -batchsize 16
        
    full training command for wiki103:

    python3 starter.py \
        -dir_name model_wiki103 \
        -savename "saved/model_wiki103/model_euclid.pth" \
        - dataset "wiki103" \
        -batchsize 16 \
        -epochs 1 \

    '''
    