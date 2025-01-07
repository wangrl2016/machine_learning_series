import torch
import numpy
from matplotlib import pyplot

def func(x):
    return 2 * numpy.sin(x) + x**0.8

class NWKernelRegression(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = torch.nn.Parameter(torch.Tensor(rng.random(1)), requires_grad=True)
    
    def forward(self, queries, keys, values):
        queries = queries.repeat_interleave(keys.shape[1]).reshape((-1, keys.shape[1]))
        self.attention_weights = torch.nn.functional.softmax(
            -((queries - keys) * self.w)** 2 / 2, dim=1)
        return torch.bmm(self.attention_weights.unsqueeze(1),
                         values.unsqueeze(-1)).reshape(-1)

if __name__ == '__main__':
    rng = numpy.random.default_rng(0)
    n_train = 50
    x_train = numpy.sort(rng.random(n_train) * 5)
    y_train = func(x_train) + rng.normal(0.0, 0.5, (n_train,))
    x_test = numpy.arange(0, 5, 0.1)
    y_test = func(x_test)
    n_test = len(x_test)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_test = torch.from_numpy(x_test)
    x_tile = x_train.repeat((n_train, 1))
    y_tile = y_train.repeat((n_train, 1))
    keys = x_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    values = y_tile[(1 - torch.eye(n_train)).type(torch.bool)].reshape((n_train, -1))
    
    net = NWKernelRegression()
    loss = torch.nn.MSELoss(reduce='none')
    optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
    
    losses = []
    for epoch in range(10):
        optimizer.zero_grad()
        ls = loss(net(x_train, keys, values), y_train)
        ls.sum().backward()
        optimizer.step()
        losses.append(round(ls.sum().item(), 4))
        # print(net.w.item())
    
    print(losses)
    pyplot.plot(numpy.arange(len(losses)), losses)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
    
    keys = x_train.repeat((n_test, 1))
    values = y_train.repeat((n_test, 1))
    y_hat = net(x_test, keys, values).unsqueeze(1).detach()
    
    pyplot.plot(x_train, y_train, 'o', alpha=0.5, label='Samples')
    pyplot.plot(x_test, y_test, label='True')
    pyplot.plot(x_test, y_hat, '--', label='Pred')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
