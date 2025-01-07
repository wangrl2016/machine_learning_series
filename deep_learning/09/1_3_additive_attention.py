import numpy
import torch

def sequence_mask(x, valid_length, value=-1e6):
    maxlen = x.size(1)
    mask = torch.arange((maxlen), device=x.device)[None, :] < valid_length[:, None]
    x[~mask] = value
    return x

def masked_softmax(x, valid_len):
    shape = x.shape
    if valid_len.ndim == 1:
        valid_len = torch.repeat_interleave(valid_len, shape[1])
    else:
        valid_len = valid_len.reshape(-1)
    x = sequence_mask(x.reshape(-1, shape[-1]), valid_len)
    return torch.nn.functional.softmax(x.reshape(shape), dim=-1)

class AdditiveAttention(torch.nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        self.W_k = torch.nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = torch.nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = torch.nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, valid_lens):
        queries, keys = self.W_q(queries), self.W_k(keys)
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
if __name__ == '__main__':
    queries, keys = torch.normal(0, 1, (2, 1, 20)), torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = AdditiveAttention(key_size=2, query_size=20, num_hiddens=8, dropout=0.1)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))
