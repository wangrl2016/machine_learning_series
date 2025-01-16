import torch
import copy

# Some convenience helper functions used throughout the notebook.

class EncoderDecoder(torch.nn.Module):
    '''
    A standard Encoder-Decoder architecture. Base for this and many other models.
    '''
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
    
    def forward(self, src, tgt, src_mask, tgt_mask):
        # Take in and process masked src and target sequences.
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(torch.nn.Module):
    # Define standard linear + softmax generation step.
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = torch.nn.Linear(d_model, vocab)
    
    def forward(self, x):
        return torch.nn.functional.log_softmax(self.proj(x), dim=-1)

def clones(module, N):
    # Produce N identical layers.
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(torch.nn.Module):
    # Core encoder is a stack of N layers.
    def __init__(self, layer, N):
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
    
    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(torch.nn.Module):
    # Construct a layernorm module (See citation for details).
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = torch.nn.Parameter(torch.ones(features))
        self.b_2 = torch.nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

if __name__ == '__main__':
    x = torch.rand(2, 3, 4)
    generator = Generator(d_model=4, vocab=10)
    probs = generator.forward(x)
    print(probs.shape)
