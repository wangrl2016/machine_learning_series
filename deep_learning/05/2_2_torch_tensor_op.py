import torch

if __name__ == '__main__':
    tensor = torch.rand(3, 4)
    # We move our tensor to the GPU if available.
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
    
    tensor = torch.ones(4, 4)
    # standard numpy-like indexing and slicing.
    tensor = torch.ones(4, 4)
    print('First row:', tensor[0])
    print('First column:', tensor[:, 0])
    print('Last column:', tensor[..., -1])
    tensor[:, 1] =0
    print(tensor)
