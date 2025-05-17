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

