import torch

if __name__ == '__main__':
    x = torch.tensor(3.0, requires_grad=True)
    y = x**2
    y.backward()
    dy_dx = x.grad
    print(dy_dx)

    w = torch.nn.Parameter(torch.full((3, 2), 0.1))
    b = torch.nn.Parameter(torch.zeros(2))
    x = torch.tensor([[1.0, 2.0, 3.0]])
    with torch.autograd.set_grad_enabled(True):
        y = torch.tanh(torch.matmul(x, w) + b)
        loss = torch.mean(y * y)
    loss.backward()
    dl_dw, dl_db = w.grad, b.grad
    print(dl_dw)
    print(dl_db)
