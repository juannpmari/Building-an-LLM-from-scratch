import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        # self.W_key = nn.Parameter(torch.rand(d_in, d_out)) #Trainable matrix
        # self.W_value = nn.Parameter(torch.rand(d_in, d_out))
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias) # nn.Linear provides better weight initialization and performs matrix multiplication (if bias=False)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # keys = x @ self.W_key
        # queries = x @ self.W_query
        # values = x @ self.W_value
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        att_weights = torch.softmax(attn_scores/ keys.shape[-1]**0.5, dim=-1)
        context_vec = att_weights @ values
        return context_vec



class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, qkv_bias=False, dropout=0.1):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer( #TODO: understand this
           'mask',
           torch.triu(torch.ones(context_length, context_length), # context_length is the length in tokens of the input sequence
           diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        #Apply causal mask and dropout
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf) #TODO: understand this
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = att_weights @ values
        return context_vec


