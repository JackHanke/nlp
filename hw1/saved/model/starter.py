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

def read_corpus(filename,tokenizer):
    seq = []
    with open(filename,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return(seq)

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x.int())

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
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
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    
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
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout, seqlen):
        super().__init__()
        self.d_model = d_model
        self.N = N
        self.seqlen = seqlen
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout, seqlen) 
        self.out = nn.Linear(d_model, trg_vocab)
    def forward(self, trg, trg_mask):
        d_output = self.decoder(trg, trg_mask)
        output = self.out(d_output)
        return output

def get_model(opt, src_vocab, trg_vocab):
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    model = Transformer(src_vocab, trg_vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout, opt.seqlen)
    model.to(opt.device)
       
    if opt.loadname is not None:
        print("loading pretrained weights...")
        model.load_state_dict(torch.load(opt.loadname))
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model

# 
class TokensDataset(Dataset):
    def __init__(self, dataset, seqlen):
        self.dataset = dataset
        self.seqlen = seqlen

    def __len__(self):
        # return -1 + len(self.dataset)//self.max_seq_length # NOTE again idk if this is right
        return len(self.dataset) - self.seqlen - 1 # NOTE again idk if this is right
        # return 15

    def __getitem__(self, idx): 
        # TODO I'm not sure which we should do, chose sliding window for now
        
        # independent chunking 
        # data = torch.LongTensor(self.dataset[idx*self.d_model:(idx+1)*self.d_model]) # sequence of context length d_model
        # label = torch.LongTensor([self.dataset[(idx+1)*self.d_model]]) # next token prediction

        # sliding window 
        data = torch.tensor(self.dataset[idx:idx+self.seqlen], dtype=torch.int64) # sequence of context length d_model
        # label = torch.tensor(self.dataset[idx+1:idx+self.seqlen+1], dtype=torch.int64) # sequence of context length d_model
        label = torch.tensor([self.dataset[idx+self.seqlen]], dtype=torch.int64) # next token prediction
        return data, label
    
def train_model(model, opt):
    model.train()
    
    #  1. create a nopeak mask
    size = model.seqlen # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)

    # 
    train_dataset = TokensDataset(dataset=opt.train, seqlen=opt.seqlen)
    
    #  2. feed training data to the model in batches
    train_loader = DataLoader(train_dataset, batch_size=opt.batchsize, shuffle=True)

    # cross entropy loss
    loss_fn = torch.nn.CrossEntropyLoss()

    training_perplexities, valid_perplexities = [], []

    best_valid_perplexity = float('inf')
    for epoch in range(opt.epochs):
        # training perplexity
        train_metric = Perplexity()
        prog_bar = tqdm(train_loader)
        for batch_index, (context_tokens, next_tokens) in enumerate(prog_bar):
            # zero gradients
            opt.optimizer.zero_grad()
            #  3. send the indices of training tokens to the GPU
            context_tokens.to(opt.device)
            next_tokens.to(opt.device)

            # inference
            predictions = model.forward(
                trg=context_tokens,
                trg_mask=nopeak_mask,
            )

            # create shifted context + next token

            # input needs to be batch by vocab by seqlen
            # target needs to be batch by seqlen

            shifted_context = torch.cat((context_tokens[:,1:], next_tokens), dim=1)
            # one hot
            # shifted_context_matrix = torch.nn.functional.one_hot(shifted_context, num_classes=opt.vocab_size)

            #  4. linearize the predictions and compute the loss against ground truth
            # loss
            train_loss_val = loss_fn(predictions.permute(0,2,1), shifted_context.long())
            # update perplexity metric
            train_metric.update(predictions, shifted_context)
            
            #  5. calculate and apply the gradients with loss.backward() and optimizer.step()
            train_loss_val.backward()
            opt.optimizer.step()

        #  6. report intermediate trainining perplexity
        training_perplexity = train_metric.compute()
        training_perplexities.append(training_perplexity)

        #  7. generate a test perplexity once per training epoch by calling test_model()
        valid_perplexity = test_model(model=model, opt=opt, epoch=epoch)
        valid_perplexities.append(valid_perplexity)

        #  8. save model weights to file specified in opt.savename
        if valid_perplexity < best_valid_perplexity:
            torch.save(model.state_dict(), opt.savename)
            best_valid_perplexity = valid_perplexity
    
    return training_perplexities, valid_perplexities
    
def test_model(model, opt, epoch):
    # write code to generate perplexity of test set
    model.eval()
    
    metric = Perplexity()

    # set progress
    if epoch >= 0:
        valid_dataset = TokensDataset(opt.valid, model.seqlen)
        val_loader = DataLoader(valid_dataset, batch_size=opt.batchsize, shuffle=False)
        prog_bar = tqdm(val_loader)
    # NOTE the starter code calls testing "epoch -1"
    elif epoch < 0:
        test_dataset = TokensDataset(opt.test, model.seqlen)
        test_loader = DataLoader(test_dataset, batch_size=opt.batchsize, shuffle=False)
        prog_bar = tqdm(test_loader)

    size = model.seqlen # get seq_len for matrix
    nopeak_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0)

    for batch_index, (context_tokens, next_tokens) in enumerate(prog_bar):
            # zero gradients
            opt.optimizer.zero_grad()
            #  3. send the indices of training tokens to the GPU
            context_tokens.to(opt.device)
            next_tokens.to(opt.device)

            # inference
            predictions = model.forward(
                trg=context_tokens,
                trg_mask=nopeak_mask,
            )

            # create shifted context + next token
            next_tokens = next_tokens
            shifted_context = torch.cat((context_tokens[:,1:], next_tokens), dim=1)
            # one hot
            shifted_context_matrix = torch.nn.functional.one_hot(shifted_context, num_classes=opt.vocab_size)

            #  4. linearize the predictions and compute the loss against ground truth
            # loss
            # test_loss_val = loss_fn(predictions, shifted_context_matrix.float())
            # update perplexity metric
            metric.update(predictions, shifted_context)

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
                
    opt = parser.parse_args()
    opt.verbose = False    
    
    # NOTE this is changed from starter because it required cuda device
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # opt.device = 0 if opt.no_cuda is False else -1
    # if opt.device == 0:
    #     assert torch.cuda.is_available()
    # opt.device = torch.device("cuda:0")
    
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
    opt.train = read_corpus('data/wiki2.train.txt',tokenizer)
    opt.valid = read_corpus('data/wiki2.valid.txt',tokenizer)
    opt.test = read_corpus('data/wiki2.test.txt',tokenizer)
    
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

    # if opt.savename is not None:
    #     try:
    #         os.mkdir(opt.savename)
    #     except:
    #         nothing = 1
    opt.src_pad = 0
    opt.trg_pad = 0
            
    train_perplexities, valid_perplexities = train_model(model,opt)
    test_perplexity = test_model(model,opt,-1)

    print(train_perplexities)
    print(valid_perplexities)

    # learning curve plotting
    plt.plot([i+1 for i in range(len(train_perplexities))],[val for val in train_perplexities],label=f'Train Perplexity')
    plt.plot([i+1 for i in range(len(valid_perplexities))],[val for val in valid_perplexities],label=f'Validation Perplexity')
    plt.legend()
    plt.ylabel(f'Perplexity')
    plt.xlabel(f'Epochs')
    plt.title(f'wiki2 Training/Validation Curves')
    plt.show()
    plt.savefig(f'./wiki2_learning_curve.png')

    print(f'Test perplexity: {test_perplexity}')
        
if __name__ == "__main__":
    main()

    # NOTE d_model is the embedding dimension

    # HACK tuah

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
        -batchsize 8
        
    full training command for wiki103:

    python3 starter.py \
        -dir_name model_wiki103 \
        -savename "saved/model_wiki103/model.pth" \
        -batchsize 8 \
        -epochs 1 \

    '''
    