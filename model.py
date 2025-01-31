import torch
from torch import tensor
from dataset import GPTTokenizer
from tqdm import tqdm
from torch.nn import functional as F

class CausalAttention(torch.nn.Module):
    def __init__(self, d_in, d_out,context_length,
                 dropout = 0, q_k_v_bias = False,
                 causal_mask = True):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.dropout = torch.nn.Dropout(dropout)
        self.causal_mask = causal_mask
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self, x):
        batch, seq_len, d_in = x.shape
        q_k_v =  torch.concat([self.W_query,self.W_key,self.W_value],dim = -1)
        # print(f"q_k_v shape: {q_k_v.shape}")
        combined_transform = x @ q_k_v
        # print(f"combined_transform shape: {combined_transform.shape}")
        query, key, value = torch.split(combined_transform,self.W_key.shape[-1], dim=2)
        attn_weights = query @ key.transpose(1,2) # (batch X seq_len X seq_len)
        if self.causal_mask:
            attn_weights = attn_weights.masked_fill(self.mask.bool()[:seq_len,:seq_len],-torch.inf)
        scaled_attn_weights = attn_weights/key.shape[-1]**0.5
        attn_scores = self.dropout(torch.softmax(scaled_attn_weights, dim=-1))
        return {"context_vectors": attn_scores @ value,
                "attn_scores": attn_scores} # context rich vectors (batch X seq_len X embedding_dim)



## applies heads sequentially
class MultiHeadAttentionWrapper(torch.nn.Module):
    def __init__(self, d_in, d_out,context_length, num_heads, dropout = 0.0, q_k_v_bias = False):
        super().__init__()
        self.out_layer = torch.nn.Parameter(torch.rand(num_heads * d_out, d_out))
        self.heads = torch.nn.ModuleList([CausalAttention(d_in, d_out,context_length) for _ in range(num_heads)])
    def forward(self, x):
        concated_context_vecs = torch.concat([head(x)['context_vectors'] for head in self.heads], dim=-1)
        return concated_context_vecs @ self.out_layer


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_in, d_out, context_length, num_heads,dropout = 0.0, causal_mask = True,
                 q_k_v_bias = False):
        super().__init__()
        self.num_heads = num_heads
        self.d_out = d_out
        self.W_query = torch.nn.Linear(d_in, num_heads * d_out, bias=q_k_v_bias)
        self.W_key = torch.nn.Linear(d_in, num_heads * d_out, bias=q_k_v_bias)
        self.W_value = torch.nn.Linear(d_in, num_heads * d_out, bias=q_k_v_bias)
        self.dropout = torch.nn.Dropout(dropout)
        self.out_project = torch.nn.Parameter(torch.rand(num_heads * d_out, d_out, dtype=torch.float32))
        self.causal_mask = causal_mask
        self.register_buffer('mask', torch.triu(torch.ones(context_length,context_length),diagonal=1))

    def forward(self, x):
        batch_size, seq_len, d_in = x.shape
        # q_k_v =  torch.concat([self.W_query,self.W_key,self.W_value],dim = -1)
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # Reshape for multi-head attention
        queries = queries.view(batch_size, seq_len, self.num_heads, self.d_out)
        keys = keys.view(batch_size, seq_len, self.num_heads, self.d_out)
        values = values.view(batch_size, seq_len, self.num_heads, self.d_out)

        # Transpose the shapes (Grouping with respect to number of heads)
        queries = queries.transpose(1,2)# (batch X num_heads X seq_len X d_out)
        keys = keys.transpose(1,2)# (batch X num_heads X seq_len X d_out)
        values = values.transpose(1,2)# (batch X num_heads X seq_len X d_out)

        # Calculating Attention Scores
        attn_weights = queries @ keys.transpose(2,3) # (batch X num_heads X seq_len X seq_len)

        # Adding the Causal Mask
        if self.causal_mask:
            attn_weights = attn_weights.masked_fill(self.mask.bool()[:seq_len,:seq_len],-torch.inf)# (batch X num_heads X seq_len X seq_len)

        # Scaling and Softmax the atten scores
        scaled_attn_weights = attn_weights/keys.shape[-1]**0.5# (batch X num_heads X seq_len X seq_len)
        attn_scores = self.dropout(torch.softmax(scaled_attn_weights, dim=-1))# (batch X num_heads X seq_len X seq_len)
        # print(f"Attention Shape::{attn_scores.shape}")
        # print(f"Max val in attn score matrix:: {attn_scores.max().detach().numpy()}")
        context_vectors = (attn_scores @ values).transpose(1,2)# (batch X seq_len X num_heads X d_out)
        concated_context_vectors = context_vectors.contiguous().view(batch_size,seq_len,-1)# (batch X seq_len X num_heads * d_out)
        context_vectors = concated_context_vectors @ self.out_project# (batch X seq_len X d_out)
        return {"context_vectors": context_vectors,
                "attn_scores": attn_scores} # context vectors (batch X seq_len X embedding_dim)
                                            # attn_scores (batch X num_heads X seq_len X seq_len)


class LayerNorm(torch.nn.Module):

    def __init__(self, embedding_dim):
        super().__init__()
        self.eps = 1e-5 # to avoid zero divison
        self.scale = torch.nn.Parameter(torch.ones(embedding_dim))
        self.shift = torch.nn.Parameter(torch.zeros(embedding_dim))

    def forward(self, x:torch.tensor):
        mean = x.mean(dim = -1, keepdim = True)
        var =  x.var(dim = -1, unbiased = False,keepdim = True) # Bessel's Correction
        normed = (x-mean)/torch.sqrt(var + self.eps)
        return (normed * self.scale) + self.shift # gives more flexibility


class GELU(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        # this formula is a simplification/approximation
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*
                                         x + 0.044715 * torch.pow(x,3)))


class FeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        embedding_dim = config.get("embedding_dim")
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 4 * embedding_dim), # moves the input to a high dimensional space
            GELU(),
            torch.nn.Linear(4 * embedding_dim, embedding_dim)# move the activation back to embedding dimension space
        )
    def forward(self, x):
        return self.layers(x)



class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["embedding_dim"],
            d_out=cfg["embedding_dim"],
            context_length=cfg["context_len"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = torch.nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x) # Shape [batch_size, num_tokens, emb_size]
        x = self.att(x)["context_vectors"]   # Shape [batch_size, num_tokens, emb_size]
        # print(type(x))
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        # Shortcut connection for feed-forward block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # Add the original input back

        return x


class GPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.emb_layer = torch.nn.Embedding(self.config.get('vocab_size'),
                                            self.config.get('embedding_dim'))
        self.pos_enc_layer = torch.nn.Embedding(self.config.get("context_len"),
                                                self.config.get("embedding_dim"))
        self.drop_emb_layer = torch.nn.Dropout(self.config.get('dropout_rate'))

        # Transformer Blocks
        self.trf_blocks = torch.nn.Sequential(*[TransformerBlock(self.config) \
                                                for _ in range(self.config.get("n_layers"))])
        # Layer Norm
        self.final_norm = LayerNorm(self.config.get("embedding_dim"))

        # Output Head
        self.out_head = torch.nn.Linear(self.config.get("embedding_dim"),
                                        self.config.get("vocab_size"))

    def forward(self, inp_):
        batch_size, seq_len = inp_.shape
        tok_emb = self.emb_layer(inp_)
        pos_emb = self.pos_enc_layer(torch.arange(seq_len, device = inp_.device))
        inp_emb = tok_emb + pos_emb
        inp_emb = self.drop_emb_layer(inp_emb)
        x = self.trf_blocks(inp_emb)
        x = self.final_norm(x)
        logits = self.out_head(x) # (batch_size X seq_len X vocab_size)
        return logits

def model_inference(model, idx, max_new_tokens, context_size, temperature = 1.0):
    '''Returns the generated token ids along with the input prompt tokens'''
# idx: batch_size X seq_len
    for  _ in range(max_new_tokens):
        idx_cond = idx[:,-context_size:]
        with torch.inference_mode():
            logits = model(idx_cond) # (batch_size x contex_len x vocab_size)
            # since we are only interested in the output of last token
            logits = logits[:,-1,:] # (batch_size x vocab_size)
            # logits --> probabilities
            logits/= temperature
            probs= torch.softmax(logits,dim=-1) # (batch_size x vocab_size)
            # tok_idx = torch.argmax(probs, dim=-1, keepdim=True) # (batch_size x 1)
            tok_idx = torch.multinomial(probs, num_samples=1)
            # print(tok_idx.shape)
        idx = torch.concat([idx,tok_idx], dim = 1)
    return idx

def generate(prompts:list,model,max_new_tokens=100,
             temp:float = 1.0, DEVICE = 'cpu'):
  tok_tensor = tensor(GPTTokenizer.encode_batch(prompts)).to(DEVICE)
  out_tokens = model_inference(model.to(DEVICE),tok_tensor, max_new_tokens=max_new_tokens,
                              context_size=256, temperature=2).detach().cpu().numpy()
  responses = GPTTokenizer.decode_batch(out_tokens)
  return {f"resp_{i}":resp for i, resp in enumerate(responses,1)}

def train_model(model,optimizer, train_loader, epochs = 10, DEVICE = 'cpu'):
    # optimizer = optim.AdamW(model.parameters(),
    #                         lr = 0.01)
    tokens_seen = 0
    losses = []
    EPOCHS = epochs
    # epoch_pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
    # epoch_pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")

    for epoch in range(EPOCHS):
        model.train()
        batch_loss = 0


        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for batch in batch_pbar:
            optimizer.zero_grad()
            out_logits = model(batch['x'].to(DEVICE))
            loss = F.cross_entropy(out_logits.flatten(0,1), batch['y'].flatten(0).to(DEVICE))
            loss.backward()
            optimizer.step()
            tokens_seen += batch['x'].numel()
            batch_loss += loss.item()


        avg_loss = batch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"EPOCH : {epoch + 1} | Epoch Loss : {losses[epoch]}")
    return model, optimizer, losses

def model_inference_stream(model, idx, max_new_tokens, context_size, temperature=1.0):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.inference_mode():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            tok_idx = torch.multinomial(probs, num_samples=1)
        idx = torch.concat([idx, tok_idx], dim=1)
        yield tok_idx.item()

def generate_stream(prompts: list, model, max_new_tokens=100, temp: float = 1.0, DEVICE='cpu'):
    tok_tensor = tensor(GPTTokenizer.encode_batch(prompts)).to(DEVICE)

    for i, prompt in enumerate(prompts, 1):
        yield f"resp_{i}", prompt  # Yield the initial prompt

        for new_token in model_inference(model.to(DEVICE), tok_tensor[i-1:i], max_new_tokens=max_new_tokens,
                                         context_size=256, temperature=temp):
            new_token_decoded = GPTTokenizer.decode([new_token])
            yield f"resp_{i}", new_token_decoded
def stream_response(prompts, model, max_new_tokens=100, temp=1.0, DEVICE='cpu'):
    for _,token in generate(prompts, model, max_new_tokens, temp, DEVICE):
        yield token