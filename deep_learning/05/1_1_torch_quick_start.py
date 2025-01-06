import torchvision
import torch

# Define model
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f'Loss: {loss:>7f} [{current:>5d}/{size:>5d}]')
    
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f'Test error:\n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}')   

if __name__ == '__main__':
    # Download training data from open datasets.
    training_data = torchvision.datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    
    # Download test data from open datasets.
    test_data = torchvision.datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor(),
    )
    
    batch_size = 64
    # Create data loaders.
    train_dataloader = torch.utils.data.DataLoader(training_data,
                                                   batch_size=batch_size)
    test_dataloader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=batch_size)
    for X, y in test_dataloader:
        print(f'Shape of X [N, C, H, W]: {X.shape}')
        print(f'Shape of y: {y.shape} {y.dtype}')
        break
    
    # Get cpu, gpu or mps device for training.
    device = (
        'cuda'
        if torch.cuda.is_available()
        else 'mps'
        if torch.backends.mps.is_available()
        else 'cpu'
    )
    print(f'Using {device} device')
    
    model = NeuralNetwork().to(device)
    print(model)
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    epochs = 10
    for t in range(epochs):
        print(f'Epoch {t+1}\n-----------------------------')
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print('Done!')
