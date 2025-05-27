import math
import copy
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch.amp import autocast, GradScaler
from transformers import GPT2TokenizerFast

from nlp_metrics import *

PAD_TOKEN_INDEX = 0

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def masked_accuracy(output: torch.Tensor, target: torch.Tensor):
    mask = target.ne(PAD_TOKEN_INDEX)
    output = output.argmax(-1).masked_select(mask)
    target = target.masked_select(mask)
    return (output == target).float().mean()

#
def test_model(model: nn.Module, test: list, masked: bool = False):
    model.eval()

    num_right = 0
    num_seen = 0
    num_total = len(test)

    validation_loss = []

    bleu, rouge, bert = 0, 0, 0

    prog_bar = tqdm(enumerate(test), total=num_total)
    for i, (x_batch, y_batch) in prog_bar:
        # send to gpu
        x_batch = x_batch.to(model.device)
        y_batch = y_batch.to(model.device)

        # make padding mask
        trg_mask = (x_batch != PAD_TOKEN_INDEX).unsqueeze(1).to(model.device)

        # make inference
        predictions = model.forward(trg=x_batch, trg_mask=trg_mask)
        
        # contigous conversion
        y_batch = y_batch.contiguous()

        # NOTE validation CEL
        valid_loss_val = F.cross_entropy(predictions.view(-1, model.src_vocab), y_batch.view(-1), ignore_index=PAD_TOKEN_INDEX)
        validation_loss.append(valid_loss_val.item())

        pred_ans_tokens = torch.argmax(predictions[:, -1], dim=-1)
        # NOTE validation accuracy
        # if next token accuracy
        if not masked:
            ans_tokens = []
            temp = y_batch != PAD_TOKEN_INDEX
            for i, row in enumerate(temp):
                valid_tokens = y_batch[i][row]
                ans_tokens.append(valid_tokens[-1])

            ans_tokens = torch.stack(ans_tokens)
 
            temp = torch.sum(torch.eq(ans_tokens, pred_ans_tokens))

            num_right += torch.sum(torch.eq(ans_tokens, pred_ans_tokens))

            num_seen += y_batch.size(0)

            prog_bar.set_description(f'Question {num_seen}: {(100*num_right/num_seen):.4f} percent. Validation Loss: {valid_loss_val:.4f}')

        # if masked accuracy
        elif masked:
            num_right += masked_accuracy(output=predictions, target=y_batch)

            for i in range(y_batch.shape[0]):
                # exmp_str = tokenizer.decode(y_batch[i], skip_special_tokens=True).replace('!', '')
                # pred_string = tokenizer.decode(pred_ans_tokens[i], skip_special_tokens=True).replace('!', '')
                exmp_str = tokenizer.decode(y_batch[i]).replace('!', '')
                pred_string = tokenizer.decode(pred_ans_tokens[i]).replace('!', '')

                # god this sucks
                min_len = min(len(exmp_str), len(pred_string))
                # print(len(exmp_str))
                # print(len(pred_string))

                bleu += compute_bleu(reference=exmp_str.split(), candidate=pred_string.split())
                rouge += compute_rouge(reference=exmp_str, candidate=pred_string)['rouge1'].fmeasure
                # bert += compute_bertscore(references=exmp_str[:min_len].split(), candidates=pred_string[:min_len].split())[2][0]

            num_seen += y_batch.size(0)
            prog_bar.set_description(f'Bleu: {(bleu/num_seen):.4f} ROUGE: {(rouge/num_seen):.4f} BERT: {(bert/num_seen):.4f}')

        bleu = bleu / num_seen
        rouge = rouge / num_seen
        bert = bert / num_seen

    if masked: return num_right/num_seen, validation_loss, bleu, rouge, bert
    return num_right/num_seen, validation_loss

# training script     
def train_model(
        model: torch.nn.Module, 
        train: torch.Tensor, 
        valid: torch.Tensor, 
        epochs: int,
        batch_size: int, 
        only_last_token: bool,
        savename: str
    ):

    # logging.basicConfig(filename='training.log', level=logging.DEBUG)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    # set model to train mode
    model.train()

    # make scaler for casting
    scaler = GradScaler()

    # training loop
    for epoch in range(epochs):

        # Get data batches
        prog_bar = tqdm(enumerate(train), total=len(train))
        
        # batch training
        for i, (x_batch, y_batch) in prog_bar:
            x_batch = x_batch.to(model.device)
            y_batch = y_batch.to(model.device)

            # make padding mask
            trg_mask = (x_batch != PAD_TOKEN_INDEX).unsqueeze(1).to(model.device)
            # print(trg_mask[0])

            # autocast for lower precision training
            with autocast(device_type=y_batch.device.type, dtype=torch.float16):
                # zero gradients
                optimizer.zero_grad()

                # inference
                predictions = model.forward(trg=x_batch, trg_mask=trg_mask)
                # contigous conversion
                y_batch = y_batch.contiguous()

                if only_last_token:
                    last_tokens_pred = predictions[:, -1, :]
                    ans_tokens = []
                    temp = y_batch != PAD_TOKEN_INDEX
                    for i, row in enumerate(temp):
                        valid_tokens = y_batch[i][row]
                        ans_tokens.append(valid_tokens[-1])

                    ans_tokens = torch.stack(ans_tokens)
                    # loss
                    train_loss_val = F.cross_entropy(last_tokens_pred, ans_tokens, ignore_index=PAD_TOKEN_INDEX)

                elif not only_last_token:
                    # loss
                    train_loss_val = F.cross_entropy(predictions.view(-1, model.src_vocab), y_batch.view(-1), ignore_index=PAD_TOKEN_INDEX)

                prog_bar.set_description(f'Loss: {train_loss_val:.6f}')

            #  5. calculate and apply the gradients with loss.backward() and optimizer.step()
            scaler.scale(train_loss_val).backward()
            scaler.step(optimizer)
            scaler.update()

        # validation
        acc, val_loss = test_model(model=model, test=valid)
        print(f'Epoch {epoch+1} validation accuracy: {acc*100:.4f}. Validation Loss: {(sum(val_loss)/len(val_loss)):.4f}')

    model.save(savename=savename)

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
        # self.pe = self.pe.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        pe = self.pe
        # if x.is_cuda:
        #     pe.cuda()
        x = x + self.pe[:, :x.size(1)]
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
    # NOTE for question 4...
    # scores = torch.cdist(q , k, p=2)
    
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
        self.src_vocab=src_vocab

        nopeak_mask = np.triu(np.ones((1, self.seqlen, self.seqlen)), k=1).astype('uint8')
        self.nopeak_mask = Variable(torch.from_numpy(nopeak_mask) == 0).to(self.device)

    def forward(self, trg, trg_mask = None, inference_mode: bool = False):
        if trg_mask is None:
            trg_mask = self.nopeak_mask
        elif trg_mask is not None:
            trg_mask = trg_mask & self.nopeak_mask
        if inference_mode:
            trg_mask = None

        # print(trg_mask[0])
        # print(f'trg_mask shape: {trg_mask.shape}')

        d_output = self.decoder(trg, trg_mask)
        output = self.out(d_output)
        return output

    def save(self, savename):
        torch.save(self.state_dict(), savename)
        print(f'Saved model as {savename}')

    @torch.no_grad()
    def decode(self, trg, debug: bool = False):
        # generation settings
        tokens_generated = 0
        max_tokens = 20
        token_idx = 0
        inference_tokens = []

        # TODO something like below is better I think
        # trg = trg[:-512].contiguous().to('cuda')
        trg = trg.contiguous().to('cuda')
        while token_idx != 50256 and tokens_generated < max_tokens:
            logits = self.forward(trg=trg, inference_mode=True)
            tokens_generated += 1
            # greedy decode
            token_idx = torch.argmax(logits[:, -1], dim=1)
            inference_tokens.append(token_idx.item())
            # find first index of first padding 
            i = 0
            while trg[i] != PAD_TOKEN_INDEX:
                i+=1
            # append
            if i == trg.shape[0]-1:
                trg = torch.cat(trg, token_idx)
                # context cutoff
                trg = trg[:self.seqlen]
            # fill padding
            else:
                trg[i] = token_idx.item()

        if debug: return inference_tokens, trg
        return inference_tokens
    
    @torch.no_grad()
    def decode_danya(self, trg, debug: bool = False):
        trg = trg.contiguous().to(self.device)

        max_tokens = 20
        inference_tokens = []
        
        for _ in range(max_tokens):
            logits = self.forward(trg=trg, inference_mode=True)
            next_token = torch.argmax(logits[:, -1], dim=-1)  # [batch]
            inference_tokens.append(next_token.item())

            # Append next token to trg
            trg = torch.cat([trg, next_token.unsqueeze(0)], dim=1)
            trg = trg[:, -self.seqlen:]  # context cuttoff

            if next_token.item() == 50256:  # EOS token for GPT2 tokenier
                break

        if debug:
            return inference_tokens, trg
        return inference_tokens
