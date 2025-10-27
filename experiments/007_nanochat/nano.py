from dataclasses import dataclass
import torch
from torch import nn

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

