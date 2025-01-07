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

if __name__ == '__main__':
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([[1, 3], [2, 4]])))
