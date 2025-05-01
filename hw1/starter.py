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

# read corpus and structure as tensor
def read_corpus(filename, tokenizer):
    seq = []
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip() 
            if line: 
                tokens = tokenizer(line)
                seq.extend(tokens['input_ids'])
    return torch.tensor(seq, dtype=torch.long)

# sequentially loop over batch 
def get_batch(data, seq_len, batch_size, device):
    n_batches = data.size(0) // (seq_len * batch_size)
    data = data[:n_batches * batch_size * seq_len]
    data = data.view(batch_size, -1)

    for i in range(0, data.size(1) - seq_len, seq_len):
        x = data[:, i:i+seq_len]
        y = data[:, i+1:i+1+seq_len]
        yield x.to(device), y.to(device)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 512, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = Variable(pe.unsqueeze(0), requires_grad=False)
        self.register_buffer('pe', pe)
        self.pe = self.pe.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe
        # if x.is_cuda:
        #     pe.cuda()
        x = x + self.pe
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
    
        self.size = d_model
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    # NOTE for question 4...
    # scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    scores = torch.cdist(q , k, p=2)
    # input(scores)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        # scores = scores.masked_fill(mask == 0, -1e9)
        scores = scores.masked_fill(mask == 0, -1e4)
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        output = self.out(concat)
    
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class CosineWithRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with restarts.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optional (default: 0)
        The minimum learning rate.

    last_epoch : int, optional (default: -1)
        The index of the last epoch.

    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 T_max: int,
                 eta_min: float = 0.,
                 last_epoch: int = -1,
                 factor: float = 1.) -> None:
        # pylint: disable=invalid-name
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super(CosineWithRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Get updated learning rate."""
        # HACK: We need to check if this is the first time get_lr() was called, since
        # we want to start with step = 0, but _LRScheduler calls get_lr with
        # last_epoch + 1 when initialized.
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lrs = [
            (
                self.eta_min + ((lr - self.eta_min) / 2) *
                (
                    np.cos(
                        np.pi *
                        ((self._cycle_counter) % self._updated_cycle_len) /
                        self._updated_cycle_len
                    ) + 1
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            # Adjust the cycle length.
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step

        return lrs

# build a decoder layer with two multi-head attention layers and
# one feed-forward layer   
# modified for decoder-only  
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, trg_mask):
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x  

# modified decoder for decoder-only
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout, seqlen):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, max_seq_len=seqlen, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, trg, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)

# modified transformer for decoder-only
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, seqlen, device):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.seqlen = seqlen
        self.device = device
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, seqlen) 
        self.out = nn.Linear(d_model, trg_vocab)

        nopeak_mask = np.triu(np.ones((1, self.seqlen, self.seqlen)), k=1).astype('uint8')
        self.nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(self.device)
    def forward(self, trg):
        d_output = self.decoder(trg, self.nopeak_mask)
        output = self.out(d_output)
        return output

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

    
def train_model(model, opt):
    model.train()

    training_perplexities, valid_perplexities = [], []

    scaler = GradScaler()

    best_valid_perplexity = float('inf')
    for epoch in range(opt.epochs):
        # training perplexity
        train_metric = Perplexity()
        train_metric.to(opt.device)

        # Get data batches
        data_iter = get_batch(opt.train, opt.seqlen, opt.batchsize, opt.device)
        prog_bar = tqdm(enumerate(data_iter))
        

        for i, (x_batch, y_batch) in prog_bar:
            # autocast for lower precision training
            with autocast(device_type=y_batch.device.type, dtype=torch.float16):
                # zero gradients
                opt.optimizer.zero_grad()

                # inference
                predictions = model.forward(trg=x_batch)

                #  4. linearize the predictions and compute the loss against ground truth
                # loss
                train_loss_val = F.cross_entropy(predictions.view(-1, opt.vocab_size), y_batch.view(-1), ignore_index=-1) # Assuming -1 is not a valid token ID

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
def test_model(model, opt, epoch):
    # write code to generate perplexity of test set
    
    model.eval()
    
    metric = Perplexity()
    metric.to(opt.device)

    if epoch >= 0:
        data_iter = get_batch(opt.valid, opt.seqlen, opt.batchsize, opt.device)
    # NOTE the starter code calls testing "epoch -1"
    elif epoch < 0:
        data_iter = get_batch(opt.test, opt.seqlen, opt.batchsize, opt.device)

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
    