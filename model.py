import torch
from torch import tensor
from dataset import GPTTokenizer
from tqdm import tqdm
from torch.nn import functional as F
from torch import nn

# class CausalAttention(torch.nn.Module):
#     def __init__(self, d_in, d_out,context_length,
#                  dropout = 0, q_k_v_bias = False,
#                  causal_mask = True):
#         super().__init__()
#         self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
#         self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
#         self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
#         self.dropout = torch.nn.Dropout(dropout)
#         self.causal_mask = causal_mask
#         self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))

#     def forward(self, x):
#         batch, seq_len, d_in = x.shape
#         q_k_v =  torch.concat([self.W_query,self.W_key,self.W_value],dim = -1)
#         # print(f"q_k_v shape: {q_k_v.shape}")
#         combined_transform = x @ q_k_v
#         # print(f"combined_transform shape: {combined_transform.shape}")
#         query, key, value = torch.split(combined_transform,self.W_key.shape[-1], dim=2)
#         attn_weights = query @ key.transpose(1,2) # (batch X seq_len X seq_len)
#         if self.causal_mask:
#             attn_weights = attn_weights.masked_fill(self.mask.bool()[:seq_len,:seq_len],-torch.inf)
#         scaled_attn_weights = attn_weights/key.shape[-1]**0.5
#         attn_scores = self.dropout(torch.softmax(scaled_attn_weights, dim=-1))
#         return {"context_vectors": attn_scores @ value,
#                 "attn_scores": attn_scores} # context rich vectors (batch X seq_len X embedding_dim)



## applies heads sequentially
# class MultiHeadAttentionWrapper(torch.nn.Module):
#     def __init__(self, d_in, d_out,context_length, num_heads, dropout = 0.0, q_k_v_bias = False):
#         super().__init__()
#         self.out_layer = torch.nn.Parameter(torch.rand(num_heads * d_out, d_out))
#         self.heads = torch.nn.ModuleList([CausalAttention(d_in, d_out,context_length) for _ in range(num_heads)])
#     def forward(self, x):
#         concated_context_vecs = torch.concat([head(x)['context_vectors'] for head in self.heads], dim=-1)
#         return concated_context_vecs @ self.out_layer


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) 
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), ## Expansion
            GELU(), ## Activation
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]), ## Contraction
        )

    def forward(self, x):
        return self.layers(x)



class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        # 2*4*768
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


def text_to_token_ids(prompts:list, tokenizer):
    encoded = tokenizer.encode_batch(prompts, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded) # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    decode = token_ids.detach().cpu().numpy()
    return tokenizer.decode_batch(decode)


def inference(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None,
              DEVICE = 'cpu'):
    model = model.to(DEVICE)
    idx = idx.to(DEVICE)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        # top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1].reshape(-1,1)
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        # temperature scaling
        if temperature > 0.0:
            logits = logits / temperature
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
        # Otherwise get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)
        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break
        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)
    return idx

def generate(prompts:list,
             context_size:int,
             model,
             tokenizer,
             temperature = 0.2,
             top_K = 50,
             max_new_tokens = 30,
             eos_id = None,
             DEVICE = 'cpu'):
    input_tokens = text_to_token_ids(prompts, GPTTokenizer)
    out_tokens = inference(model =model,
                           idx = input_tokens,
                           context_size=context_size,
                           temperature = temperature,
                           top_k = top_K,
                           max_new_tokens=max_new_tokens,
                           DEVICE=DEVICE,
                           eos_id=eos_id)
    return token_ids_to_text(out_tokens, tokenizer=tokenizer)