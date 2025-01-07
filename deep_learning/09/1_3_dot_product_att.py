import math
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

class DotProductAttention(torch.nn.Module):
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = torch.nn.Dropout(dropout)
    
    def forward(self, queries, keys, values, valid_lens=None):
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
if __name__ == '__main__':
    queries = torch.normal(0, 1, (2, 1, 2))
    keys = torch.ones((2, 10, 2))
    values = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4).repeat(2, 1, 1)
    valid_lens = torch.tensor([2, 6])
    attention = DotProductAttention(dropout=0.5)
    attention.eval()
    print(attention(queries, keys, values, valid_lens))
        