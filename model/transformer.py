import torch 
import torch.nn as nn 
import math


class InputEmbeddings(nn.Module):
    def __init__(self, channel_in: int, d_model: int):
        super().__init__()
        self.channel_in = channel_in
        self.d_model = d_model
        self.embedding = nn.Linear(channel_in, d_model)

    def forward(self, x, mask): 
        # x: [Batch, seq_len, input_dim] -> [Batch, seq_len, d_model] 
        if mask is not None:
            x = x.masked_fill(mask==0, 1e-9)
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len 
        self.dropout = nn.Dropout(p=dropout)

        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vector of shape (seq_len, 1)
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(-1) 
        _2i = torch.arange(0, d_model, step=2, dtype=torch.float)  # (d_model // 2, )
        pe[:, 0::2] = torch.sin(position / 10000 ** (_2i / self.d_model))
        pe[:, 1::2] = torch.cos(position / 10000 ** (_2i / self.d_model))

        pe = pe.unsqueeze(0)  #  (1, Seq_Len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x): 
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module): 
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps 
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added 

    def forward(self, x): 
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias 


class FeedForwardBlock(nn.Sequential):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None: 
        layers = list() 
        layers.append(nn.Linear(d_model, d_ff))  # W1 and B1
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(d_ff, d_model))  # W2 and B2 
        super().__init__(*layers)


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__() 
        assert d_model % num_heads == 0, "d_model is not divisible by number of heads"
        self.d_model = d_model
        self.num_heads = num_heads 

        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq 
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = FeedForwardBlock(d_model, d_model, dropout)
        
        self.dropout = nn.Dropout(p=dropout)
    
    @staticmethod
    def attention(key, query, value, dropout: nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) /math.sqrt(d_k) # (B, seq_len, d_k) @ (B, d_k, seq_len) -> (B, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)
        if dropout is not None: 
            attention_scores = dropout(attention_scores)  
        
        return (attention_scores @ value), attention_scores

    def forward(self, k, q, v): 
        key = self.w_k(k)  # (B, Seq_len, d_model) -> (B, seq_len, d_model)
        query = self.w_q(q) # (B, Seq_len, d_model) -> (B, seq_len, d_model)
        value = self.w_v(v) # (B, Seq_len, d_model) -> (B, seq_len, d_model)

        # (Batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)
           
        x, self.attention_scores = MultiHeadAttentionBlock.attention(key, query, value, self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.num_heads * self.d_k) # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (batch, seq_len, d_model)

        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module): 
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer)


class TemporalLayer(nn.Module):
    def __init__(self, channel_in: int, d_model: int, d_ff: int, seq_len: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.channel_in = channel_in
        self.d_model = d_model 
        self.d_ff = d_ff
        self.seq_len = seq_len 
        self.num_heads = num_heads
        self.dropout = dropout 

        self.emb = InputEmbeddings(channel_in, d_model)
        self.pe = PositionalEncoding(d_model, seq_len, dropout)
        self.multi_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.rc = ResidualConnection(dropout)
        self.ln = LayerNormalization()
        self.w_o = nn.Linear(d_model, d_model)  # Wo
    
    def forward(self, x, mask): 
        # x: [Batch, Seq_len, Channel_in]
        emb = self.pe(self.emb(x, mask))
        att = self.multi_attention(emb, emb, emb)
        emb = self.ln(self.w_o(self.rc(emb, att)))
        return emb 


if __name__ == '__main__':
    input_tensor = torch.randn(size=(64, 20, 89, 9))
    te = TemporalLayer(9, 256, 256, 20, 4, 0.3)
    mask = torch.ones(19)
    mask = torch.concatenate([mask, torch.zeros(1)]).unsqueeze(-1).unsqueeze(0)
    out = torch.empty(size=(input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2], 256))
    for i in range(input_tensor.shape[2]):
        out[:, :, i, :] = te(input_tensor[:, :, i, :], mask)
    print(out.shape)


