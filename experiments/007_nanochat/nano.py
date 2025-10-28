from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

"""
This file is me reimplementing Karpathy's nanochat repo while reducing it to the basics for inference.
Explaining some of the details along the way for self-study.

Some differences from this and the original:
- Tiktoken tokenizer instead of his rust bpe implementation.
- Not worried about special tokens / tool calling scaffolding
- No KV cache
- No RoPE for rotary embeddings

Summarized architecture and components:
- Tokenizer, text to tokens
- Embedding layer, tokens to embeddings
- Transformer blocks
- Transformer block consists of: attention, feed-forward, residual connections
- Layer normalization, after each transformer block
- Softmax layer, embeddings to probabilities, and sampling
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
    This is hardcoded for equal heads right now.
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

        # gqa false cause n_head == n_kv_head, would need different impl if not equal
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=False)
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
        return x

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            # word token embeddings
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            # positional embeddings
            "wpe": nn.Embedding(config.sequence_len, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
    def forward(self, idx):
        B, T = idx.size() # batch size, sequence length

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        # use basic positional embeddings to simplify in absece of RoPE/rotary
        x = self.transformer.wte(idx) + self.transformer.wpe(pos)
        x = norm(x)
        for block in self.transformer.h:
            # kv_cache would cache computation for attention here
            # nanoChat also uses RoPE for rotary embeddings, encodes positional information at the attention layer
            # Taking out to simplify, focusing on transformer arch
            x = block(x)
        x = norm(x)
        logits = self.lm_head(x)

        # Neural Combinatorial Optimization Bello et al 2017, and Gemma 2 2024 describe clipping or softcapping logits to improve exploration + stability"
        # This effectively limits logits to +/- softcap
        softcap = 15
        logits = softcap * torch.tanh(logits / softcap)
        return logits

def sample_next_token(logits, temperature, rng=None, top_k=None):
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
        k = min(top_k, logits.size(-1))
        # grab top k logits by value
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        # sample based on the probabilities, allows for exploration
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        # gather idx choice along column 1
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


if __name__ == "__main__":
    config = GPTConfig()

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
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    tokens = tokenizer.encode(input)
    config.vocab_size = tokenizer.n_vocab
    print("Type of tokens:", type(tokens))
    print("Tokens:", tokens)
    tokens = torch.tensor([tokens])
    gpt = GPT(config)

    """
    Nanochat Engine wrapper around core model handles initialization,kv cache, code+assistant special tokens and state tracking.
    Let's focus on the core model for now and reduce it to just gathering next tokens.
    """
    result = []
    # max reply length 10 tokens
    for _ in range(10):
        logits = gpt(tokens)
        # Softcapped non-softmaxed logits. Depending on temperature, topk ect. different ways to sample next token
        logits = logits[:, -1, :] # (B, vocab_size) we predict next token for each position in sequence, but we only look at the last position aka next token
        temperature = 1.0
        next_token = sample_next_token(logits, temperature, rng=None, top_k=None)
        #if next_token.item() == tokenizer.is_special_token:
        # not looking up what the specific tokenizers value is this is dummy
        if next_token.item() == "<|eos|>":
            break
        result.append(next_token.item())
        tokens = torch.cat([tokens, next_token], dim=1)

    response = tokenizer.decode(result)
    print(response)

