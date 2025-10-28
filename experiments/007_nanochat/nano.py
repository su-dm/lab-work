from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

"""
Summarized architecture and components:
- Tokenizer, text to tokens
- Embedding layer, tokens to embeddings
- Add positional encoding information
- Transformer blocks, multiple blocks
- Transformer block consists of: self-attention, feed-forward, residual connections
- Layer normalization, after each transformer block
- Softmax layer, embeddings to probabilities
"""

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (MQA)
    n_embd: int = 768

def norm(x):
    return F.rms_norm(x, (x.size(-1),))

class CausalSelfAttention(nn.Module):
    """
    Multi-Query Attention (MQA) comes from 'Fast Transformer Decoding' Shazeer et al. 2019
    Idea is instead of multiple heads for qkv use only multiple q. However Karpathy sets defautl equal heads for both.
    'GQA' Ainslie et al. 2023 shows group query attention. It's a middle ground 1<n_kv_head<n_head.
    It's a trade off between accuracy and speed. Also memory bandwidth.
    """
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=False)
    
    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # rotarty embedding would go here first
        q, k = norm(q), norm(k)
        # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # gqa false cause n_head == n_kv_head
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, gqa=False)
        # Re-assemble the heads side-by-side and project back to residual
        y = y.transpose(1,2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # convolution fully connected, 3-4x embedding size standard for allowing patterns to form
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.c_fc(x)
        # square relu showed results in 'Primer' So et al. 2021, modification to og transformer
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
    
    def forward(self, x):
        "normalize input, apply attention, residual added and normalized again, mlp on that and out with residual added"
        x = x + self.attn(norm(x))
        x = x + self.mlp(norm(x))

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            # word token embeddings
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
    
    def forward(self, idx):
        B, T = idx.size() # batch size, sequence length

        x = self.transformer.wte(idx) 
        x = norm(x)
        for block in self.transformer.h:
            # kv_cache would cache computation for attention here
            # nanoChat also uses RoPE for rotary embeddings, encodes positional information at the attention layer
            # Taking out to simplify, focusing on transformer arch
            x = block(x)
        x = norm(x)

if __name__ == "__main__":
    config = GPTConfig()
    gpt = GPT(config)

    input = "Hello, how are you?"
    """
    BPE Tokenizer is trained on text to greedily increment from character split to larger substrings
    Based off a configured vocabulary size.
    Special tokens are included in some vocabs.
    SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
    # every document ends with the End of Sequence (EOS) token that delimits documents
    "<|eos|>",
    ]
    Going to use tiktoken tokenizer for this because not focused on tokenizer implementation here.
    """
    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = tokenizer.encode(input)
    print("Type of tokens:", type(tokens))
    print("Tokens:", tokens)

    """
    Nanochat Engine wrapper around core model handles initialization,kv cache, code+assistant special tokens and state tracking.
    Let's focus on the core model for now and reduce it to just gathering next tokens.
    """




    torch.cuda.synchronize()
    result = []

    out = model.generate(tokens)
    tokenizer.decode(out)

