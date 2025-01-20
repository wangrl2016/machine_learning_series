import torch
from matplotlib import pyplot
import numpy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import pos_peg

# 文本处理：将文本转为索引，并填充到固定长度
def encode_texts(texts, word2idx, max_len):
    encoded = [
        [word2idx[word] for word in text.lower().split() if word in word2idx] 
        for text in texts
    ]
    padded = [seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len] for seq in encoded]
    return torch.tensor(padded, dtype=torch.long)

# 定义模型
class RNNClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.rnn = torch.nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.dense = torch.nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        _, hidden = self.rnn(embedded)
        # 取最后一个隐藏状态
        hidden = hidden[-1]
        out = self.dense(hidden)
        return out

if __name__ == '__main__':
    train_texts = pos_peg.train_data.keys()
    train_labels = list(map(lambda x: 1 if x else 0, pos_peg.train_data.values()))
    test_texts = pos_peg.test_data.keys()
    test_labels = list(map(lambda x: 1 if x else 0, pos_peg.test_data.values()))
    # 构建词汇表
    vocab = set(word for text in train_texts for word in text.lower().split())
    word2idx = {word: idx + 1 for idx, word in enumerate(vocab)}
    word2idx["<PAD>"] = 0  # 用0作为填充符
    print(word2idx)

    # 超参数
    max_len = 10  # 固定句子的最大长度
    embedding_dim = 64
    hidden_dim = 128
    batch_size = 2
    num_epochs = 100

    x_train = encode_texts(train_texts, word2idx, max_len)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    x_test = encode_texts(test_texts, word2idx, max_len)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    model = RNNClassifier(len(word2idx), embedding_dim, hidden_dim, output_dim=2)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            l = loss(outputs, y_batch)
            l.backward()
            optimizer.step()
            total_loss += l.item()
        losses.append(total_loss / len(train_loader))
        if epoch % 10 == 0:
            print(f'Epoch {epoch + 1}/{num_epochs}, loss: {total_loss / len(train_loader):.4f}')

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            outputs = model(x_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    
    print(f'Test accuracy: {correct / total:.4f}')
    
    pyplot.plot(numpy.array(range(len(losses))) * 50, losses, label='Loss')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

