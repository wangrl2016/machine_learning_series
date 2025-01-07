import numpy
import torch

if __name__ == '__main__':
    X = torch.ones((2, 1, 4))
    Y = torch.ones((2, 4, 6))
    output = torch.bmm(X, Y)
    assert output.shape == (2, 1, 6)

    weights = torch.ones((2, 10)) * 0.1
    values = torch.arange(20.0).reshape((2, 10))
    weights_output = torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
    assert numpy.allclose(weights_output.numpy(), [[[4.5]], [[14.5]]])
