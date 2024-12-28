import torch

class TransformerWithAttention(torch.nn.Module):
    def __init__(self):
        pass

if __name__ == '__main__:':
    vocab = {
        '<pad>': 0,
        '<sos>': 1,
        '<eos>': 2,
        'I': 3,
        'am': 4,
        'learning': 5,
        'transformer': 6,
    }
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    # model hyperparameters
    vocab_size = len(vocab)
    embed_size = 16
    num_heads = 2
    num_layers = 2
    hidden_size = 64
    max_len = 10



