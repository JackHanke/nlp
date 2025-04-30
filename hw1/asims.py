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

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from transformers import GPT2TokenizerFast

# --- Helper Functions ---

def read_corpus(filename, tokenizer):
    """Reads a text file and returns a list of token IDs."""
    seq = []
    with open(filename, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip() # Remove leading/trailing whitespace including newlines
            if line: # Process non-empty lines
                tokens = tokenizer(line)
                seq.extend(tokens['input_ids'])
    print(f"Read {len(seq)} tokens from {filename}")
    return torch.tensor(seq, dtype=torch.long) # Return as tensor

def create_causal_mask(size, device):
    """Creates a boolean upper triangular mask for causal attention."""
    # True values indicate positions to be masked (filled with -inf)
    # We want to mask positions where j > i (upper triangle, excluding diagonal)
    mask = torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)
    # Shape [size, size], True for upper triangle (j > i)
    return mask # No need to move to device again

def get_batch(data, seq_len, batch_size, device):
    """Generates batches of data for training or evaluation."""
    # Line Change: Added this function for data batching (Step 1)
    n_batches = data.size(0) // (seq_len * batch_size)
    data = data[:n_batches * batch_size * seq_len] # Trim excess data
    data = data.view(batch_size, -1) # Reshape into batches

    for i in range(0, data.size(1) - seq_len, seq_len):
        x = data[:, i:i+seq_len]
        y = data[:, i+1:i+1+seq_len] # Target is shifted by one
        yield x.to(device), y.to(device)


# --- Model Components ---

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        # Line Change: Removed .int() conversion as input should already be long tensor
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 4096, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension
        self.register_buffer('pe', pe) # Register as buffer, not parameter
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        # pe shape: [1, max_seq_len, d_model]
        # Line Change: Simplified positional encoding addition
        x = x * math.sqrt(self.d_model) # Scale embeddings
        seq_len = x.size(1)
        # Use Variable is deprecated, directly use the buffer
        # No need for requires_grad=False, buffers don't require gradients
        # No need for .cuda() check, handled by .to(device) on model
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)

class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        # Line Change: Corrected normalization formula application
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        norm = self.alpha * (x - mean) / (std + self.eps) + self.bias
        return norm

def attention(q, k, v, d_k, mask=None, dropout=None):
    # q, k, v shape: [bs, heads, seq_len, d_k]
    # scores shape: [bs, heads, seq_len, seq_len]
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        # mask shape [seq_len, seq_len] (boolean, True where to mask)
        # We need to broadcast mask to scores shape [bs, heads, seq_len, seq_len]
        # Unsqueeze mask to [1, 1, seq_len, seq_len]
        broadcast_mask = mask.unsqueeze(0).unsqueeze(0)
        # masked_fill fills elements of 'scores' with '-inf' where 'broadcast_mask' is True.
        scores = scores.masked_fill(broadcast_mask, float('-inf'))
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    # output shape: [bs, heads, seq_len, d_k]
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
        # q, k, v shape: [bs, seq_len, d_model]
        bs = q.size(0)
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_k
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        
        # calculate attention
        # Line Change: Pass the mask correctly to the attention function
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        # Line Change: Corrected view dimensions
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        # output shape: [bs, seq_len, d_model]
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# Line Change: Removed EncoderLayer class (Step 1)
# class EncoderLayer(nn.Module): ...

# Line Change: Modified DecoderLayer to remove cross-attention (Step 1)
class DecoderLayer(nn.Module):
    """ GPT-style Decoder Layer (Self-Attention + FeedForward) """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        # Line Change: Removed norm_2 for cross-attention
        self.norm_3 = Norm(d_model) # Renamed from norm_3 for clarity
        
        self.dropout_1 = nn.Dropout(dropout)
        # Line Change: Removed dropout_2 for cross-attention
        self.dropout_3 = nn.Dropout(dropout) # Renamed from dropout_3 for clarity
        
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout) # Self-attention
        # Line Change: Removed attn_2 (cross-attention)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, trg_mask):
        # x shape: [bs, seq_len, d_model]
        # trg_mask shape: [seq_len, seq_len]

        # --- Self-Attention Block ---
        x2 = self.norm_1(x)
        # Apply self-attention (q=x2, k=x2, v=x2) with causal mask
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))

        # Line Change: Removed Cross-Attention Block

        # --- FeedForward Block ---
        x2 = self.norm_3(x) # Use the renamed norm
        x = x + self.dropout_3(self.ff(x2)) # Use the renamed dropout
        return x

# Line Change: Removed Encoder class (Step 1)
# class Encoder(nn.Module): ...

# Line Change: Modified Decoder class to use modified DecoderLayer and remove encoder dependencies (Step 1)
class Decoder(nn.Module):
    """ Stack of GPT-style Decoder Layers """
    def __init__(self, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        # Embedder and PositionalEncoder will be handled in the main GPT2Model class
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, x, trg_mask):
        # x shape: [bs, seq_len, d_model] (after embedding and positional encoding)
        # trg_mask shape: [seq_len, seq_len]
        for i in range(self.N):
            x = self.layers[i](x, trg_mask)
        return self.norm(x)

# Line Change: Replaced Transformer class with GPT2Model (Step 1)
class GPT2Model(nn.Module):
    """ GPT2-style Autoregressive Language Model """
    def __init__(self, vocab_size, d_model, N, heads, dropout, tied_weights=True):
        super().__init__()
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.decoder = Decoder(d_model, N, heads, dropout) # Use the modified Decoder stack
        self.out = nn.Linear(d_model, vocab_size)

        # Line Change: Implement weight tying (Step 1 & 2)
        if tied_weights:
            if d_model != self.embed.embed.weight.size(1):
                raise ValueError(f"d_model ({d_model}) must equal embedding dim ({self.embed.embed.weight.size(1)}) for tied weights")
            print("Tying embedding and output layer weights.")
            self.out.weight = self.embed.embed.weight # Tie weights

    def forward(self, trg, trg_mask):
        # trg shape: [bs, seq_len]
        # trg_mask shape: [seq_len, seq_len]
        
        x = self.embed(trg) # [bs, seq_len, d_model]
        x = self.pe(x)      # [bs, seq_len, d_model]
        
        # Pass through decoder stack
        # Line Change: Pass only x and trg_mask to the modified decoder
        d_output = self.decoder(x, trg_mask) # [bs, seq_len, d_model]
        
        # Final output layer
        output = self.out(d_output) # [bs, seq_len, vocab_size]
        return output

# Line Change: Modified get_model to use GPT2Model and handle tying (Step 1 & 2)
def get_model(opt):
    
    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    print(f"Building GPT2Model: vocab_size={opt.vocab_size}, d_model={opt.d_model}, N={opt.n_layers}, heads={opt.heads}, dropout={opt.dropout}, tied={opt.tied}")
    
    # Pass vocab_size only once, tied parameter controls weight sharing
    model = GPT2Model(opt.vocab_size, opt.d_model, opt.n_layers, opt.heads, opt.dropout, tied_weights=opt.tied)
    model.to(opt.device)
       
    if opt.loadname is not None:
        print(f"Loading pretrained weights from {opt.loadname}...")
        try:
            model.load_state_dict(torch.load(opt.loadname, map_location=opt.device))
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Initializing weights randomly.")
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
    else:
        print("Initializing weights randomly.")
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
    
    return model

# --- Training and Evaluation ---

# Line Change: Implemented train_model (Step 1 & 2)
def train_model(model, opt):
    
    print("Starting training...")
    model.train()
    optimizer = opt.optimizer # Get optimizer from opt
    # scheduler = opt.sched # Optional scheduler

    start_time = time.time()
    total_loss = 0
    log_tokens = 0
    processed_batches = 0
    
    train_losses = []
    val_losses = [] # To store validation losses per epoch for learning curve

    for epoch in range(opt.epochs):
        epoch_start_time = time.time()
        epoch_loss = 0
        epoch_tokens = 0
        
        # Create causal mask once per epoch (if seqlen is fixed)
        causal_mask = create_causal_mask(opt.seqlen, opt.device)

        # Get data batches
        data_iter = get_batch(opt.train_data, opt.seqlen, opt.batchsize, opt.device)

        for i, (x_batch, y_batch) in enumerate(data_iter):
            # x_batch, y_batch shape: [batchsize, seqlen]
            
            optimizer.zero_grad()
            
            # Forward pass
            # Line Change: Pass causal_mask to the model
            output = model(x_batch, causal_mask) # output shape: [bs, seqlen, vocab_size]
            
            # Calculate loss
            # Reshape for cross_entropy: output -> [bs*seqlen, vocab_size], y_batch -> [bs*seqlen]
            loss = F.cross_entropy(output.view(-1, opt.vocab_size), y_batch.view(-1), ignore_index=-1) # Assuming -1 is not a valid token ID
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.norm) # Gradient clipping
            optimizer.step()
            # if opt.SGDR: scheduler.step() # Optional scheduler step

            batch_loss = loss.item()
            total_loss += batch_loss
            epoch_loss += batch_loss
            
            # Calculate tokens processed in this batch
            num_tokens = y_batch.ne(-1).sum().item() # Count non-ignored tokens
            log_tokens += num_tokens
            epoch_tokens += num_tokens
            processed_batches += 1

            # Logging
            if (i + 1) % opt.printevery == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / processed_batches # Use processed_batches for avg loss calculation
                ppl = math.exp(avg_loss) if avg_loss < 30 else float('inf') # Perplexity
                tokens_per_sec = log_tokens / elapsed if elapsed > 0 else 0
                
                print(f'Epoch: {epoch+1:02d} | Batch: {i+1:05d} | lr: {optimizer.param_groups[0]["lr"]:.5f} | '
                      f'Loss: {avg_loss:.3f} | PPL: {ppl:.2f} | Tokens/sec: {tokens_per_sec:.0f}')
                
                # Reset counters for the next logging interval, but keep epoch totals
                # total_loss = 0 # Don't reset total_loss, use epoch_loss for epoch avg
                # log_tokens = 0 # Don't reset log_tokens, use epoch_tokens
                # start_time = time.time() # Reset timer for interval speed
                # processed_batches = 0 # Don't reset processed_batches

        # --- End of Epoch ---
        epoch_duration = time.time() - epoch_start_time
        avg_epoch_loss = epoch_loss / (i + 1) # Average loss over batches in epoch
        epoch_ppl = math.exp(avg_epoch_loss) if avg_epoch_loss < 30 else float('inf')
        train_losses.append(epoch_ppl) # Store training perplexity for learning curve

        print("-" * 89)
        print(f'End of Epoch: {epoch+1:02d} | Time: {epoch_duration:.2f}s | '
              f'Avg Train Loss: {avg_epoch_loss:.3f} | Train PPL: {epoch_ppl:.2f}')
        
        # Evaluate on validation set
        val_ppl = test_model(model, opt, opt.valid_data, "Validation") # Pass validation data
        val_losses.append(val_ppl) # Store validation perplexity
        
        print(f'Validation PPL: {val_ppl:.2f}')
        print("-" * 89)

        # Save the model
        if opt.savename:
            save_path = os.path.join(opt.dir_name, f"{opt.savename}_epoch_{epoch+1}.pth")
            print(f"Saving model checkpoint to {save_path}")
            torch.save(model.state_dict(), save_path)
            # Save learning curves
            curves_path = os.path.join(opt.dir_name, f"{opt.savename}_learning_curves.pkl")
            with open(curves_path, 'wb') as f:
                pickle.dump({'train_ppl': train_losses, 'val_ppl': val_losses}, f)

        model.train() # Ensure model is back in training mode

    print("Training finished.")
    # Save final learning curves
    if opt.savename:
        curves_path = os.path.join(opt.dir_name, f"{opt.savename}_learning_curves_final.pkl")
        with open(curves_path, 'wb') as f:
            pickle.dump({'train_ppl': train_losses, 'val_ppl': val_losses}, f)
        print(f"Final learning curves saved to {curves_path}")


# Line Change: Implemented test_model (Step 1 & 2)
def test_model(model, opt, data_source, name="Test"):
    """Evaluates the model on the given data source."""
    print(f"Evaluating model on {name} set...")
    model.eval() # Set model to evaluation mode
    
    total_loss = 0
    total_tokens = 0
    
    # Create causal mask
    causal_mask = create_causal_mask(opt.seqlen, opt.device)
    
    # Get data batches
    data_iter = get_batch(data_source, opt.seqlen, opt.batchsize, opt.device) # Use same batching logic

    with torch.no_grad(): # Disable gradient calculation
        for i, (x_batch, y_batch) in enumerate(data_iter):
            output = model(x_batch, causal_mask)
            loss = F.cross_entropy(output.view(-1, opt.vocab_size), y_batch.view(-1), ignore_index=-1)
            
            num_tokens = y_batch.ne(-1).sum().item()
            total_loss += loss.item() * num_tokens # Weight loss by number of tokens
            total_tokens += num_tokens

    model.train() # Set model back to training mode
    
    if total_tokens == 0:
        print(f"Warning: No tokens processed during {name} evaluation.")
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss) if avg_loss < 30 else float('inf')
    print(f"{name} Evaluation Complete: Avg Loss: {avg_loss:.4f}, Perplexity: {perplexity:.4f}")
    return perplexity

# --- Main Execution ---

def main():
    
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-no_cuda', action='store_true', help='Disable CUDA')
    # parser.add_argument('-SGDR', action='store_true', help='Use Cosine Annealing scheduler') # Keep if needed
    parser.add_argument('-epochs', type=int, default=20, help='Number of training epochs') # Step 2 requirement
    parser.add_argument('-d_model', type=int, default=512, help='Model dimension') # Step 2 requirement
    parser.add_argument('-n_layers', type=int, default=6, help='Number of decoder layers') # Step 2 requirement
    parser.add_argument('-heads', type=int, default=8, help='Number of attention heads') # Step 2 requirement
    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout rate') # Step 2 requirement
    parser.add_argument('-batchsize', type=int, default=16, help='Batch size (adjust based on GPU memory)') # Adjusted default
    parser.add_argument('-printevery', type=int, default=100, help='Log frequency')
    parser.add_argument('-lr', type=float, default=0.0001, help='Learning rate') # Adjusted default
    parser.add_argument('-seqlen', type=int, default=512, help='Sequence length') # Step 2 requirement
    # parser.add_argument('-threshold', type=int, default=3) # Not used currently
    parser.add_argument('-savename', type=str, default='gpt2_wikitext2', help='Base name for saving model files') # Default save name
    parser.add_argument('-loadname', type=str, help='Path to load pretrained model weights')    
    parser.add_argument('-tied', type=int, default=1, choices=[0, 1], help='Tie embedding and output weights (1=True, 0=False)') # Step 2 requirement (defaulted to True)
    parser.add_argument('-dir_name', type=str, default='model_saves', help='Directory to save models and logs') # Changed default
    parser.add_argument('-norm', type=float, default=1.0, help='Gradient clipping norm') # Adjusted default
    parser.add_argument('-dataset', type=str, default='wikitext-2', choices=['wikitext-2', 'wikitext-103'], help='Dataset to use') # For Step 2 & 3

    opt = parser.parse_args()
    
    # --- Setup Device ---
    opt.device = torch.device("cuda" if not opt.no_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {opt.device}")
    if opt.device == torch.device("cuda"):
        torch.cuda.manual_seed(10)

    # --- Setup Logging and Saving ---
    time_name = time.strftime("%y%m%d_%H%M%S")
    opt.time_name = time_name
    # Line Change: Simplified directory creation
    dir_name = f"saved/{opt.dir_name}_{opt.dataset}_{time_name}"
    opt.dir_name = dir_name # Store full path in opt
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    # Copy script for reproducibility
    try:
        source_name = sys.argv[0]
        shutil.copy(source_name, os.path.join(dir_name, os.path.basename(source_name)))
    except Exception as e:
        print(f"Warning: Could not copy script: {e}")
        
    opt.log_file = os.path.join(dir_name, "log_file.txt")
    # Redirect print to log file (optional)
    # sys.stdout = open(opt.log_file, 'w') 
    
    print("Options:")
    print(str(opt))
    
    # --- Load Tokenizer and Data ---
    print("Loading tokenizer...")
    # Line Change: Use cache_dir to potentially avoid re-downloading
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", cache_dir="./hf_cache")
    opt.vocab_size = tokenizer.vocab_size # Get vocab size from tokenizer
    print(f"Vocabulary Size: {opt.vocab_size}")

    print(f"Loading dataset: {opt.dataset}...")
    if opt.dataset == 'wikitext-2':
        opt.train_data = read_corpus('wiki2.train.txt', tokenizer)
        opt.valid_data = read_corpus('wiki2.valid.txt', tokenizer)
        opt.test_data = read_corpus('wiki2.test.txt', tokenizer)
    elif opt.dataset == 'wikitext-103':
        # Placeholder for Step 3
        print("Loading Wikitext-103 (ensure files exist: wiki103.train.txt, etc.)")
        opt.train_data = read_corpus('wiki103.train.txt', tokenizer)
        opt.valid_data = read_corpus('wiki103.valid.txt', tokenizer)
        opt.test_data = read_corpus('wiki103.test.txt', tokenizer)
        # Adjust epochs for Step 3 if needed (e.g., set opt.epochs = 1 if dataset is wikitext-103)
        if opt.epochs != 1:
             print(f"Warning: For Wikitext-103 (Step 3), typically only 1 epoch is required. Current setting: {opt.epochs} epochs.")
             # You might want to force epochs to 1 here:
             # opt.epochs = 1
    else:
        raise ValueError("Invalid dataset specified")

    print("Dataset loading complete.")
    
    # --- Build Model ---
    model = get_model(opt) # Pass opt object
        
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])        
    text = f'Total Trainable Parameters: {params:,}' # Formatted number
    print(text)
    with open(opt.log_file, 'a') as f: # Append to log file
        f.write(text + '\n')

    # --- Setup Optimizer ---
    opt.optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-9)
    # Optional Scheduler Setup
    # if opt.SGDR:
    #     opt.train_len = len(list(get_batch(opt.train_data, opt.seqlen, opt.batchsize, opt.device))) # Estimate batches for scheduler
    #     opt.sched = CosineWithRestarts(opt.optimizer, T_max=opt.train_len)
    # else:
    #     opt.sched = None

    # --- Train and Evaluate ---
    try:
        train_model(model, opt) # Pass opt object
        print("\n--- Final Test Evaluation ---")
        final_test_ppl = test_model(model, opt, opt.test_data, "Test") # Pass test data
        print(f"Final Test Perplexity: {final_test_ppl:.4f}")
        with open(opt.log_file, 'a') as f:
             f.write(f"\nFinal Test Perplexity: {final_test_ppl:.4f}\n")

    except KeyboardInterrupt:
        print("Training interrupted by user.")
    except Exception as e:
        print(f"An error occurred during training/evaluation: {e}")
        import traceback
        traceback.print_exc()
        
    print("Script finished.")
    # Close log file if redirecting stdout
    # if isinstance(sys.stdout, file):
    #    sys.stdout.close()

if __name__ == "__main__":
    main()
