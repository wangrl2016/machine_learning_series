import numpy
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import pos_peg
from matplotlib import pyplot

def softmax(xs):
    # Applies the softmax function to the input array.
    return numpy.exp(xs) / sum(numpy.exp(xs))

class RNN:
    # A vanilla recurrent neural network.
    def __init__(self, input_size, output_size, hidden_size=64):
        rng = numpy.random.default_rng(seed=0)
        self.Whh = rng.standard_normal((hidden_size, hidden_size)) / 1000
        self.Wxh = rng.standard_normal((hidden_size, input_size)) / 1000
        self.Why = rng.standard_normal((output_size, hidden_size)) / 1000
        self.bh = numpy.zeros((hidden_size, 1))
        self.by = numpy.zeros((output_size, 1))
    
    def forward(self, inputs):
        h = numpy.zeros((self.Whh.shape[0], 1))

        self.last_inputs = inputs
        self.last_hs = { 0: h}

        # Perform each step of the RNN.
        for i, x in enumerate(inputs):
            h = numpy.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self.last_hs[i + 1] = h
        # Compute the output.
        y = self.Why @ h + self.by
        return y, h
    
    def backprop(self, dy, learn_rate=2e-2):
        # dy (dl/dy) has shape (output_size, 1)
        n = len(self.last_inputs)

        # calculate dl/dwhy and dl/dby
        dwhy = dy @ self.last_hs[n].T
        dby = dy
        
        # initialize dl/dwhh, dl/dwxh, and dl/dbh to zero
        dwhh = numpy.zeros(self.Whh.shape)
        dwxh = numpy.zeros(self.Wxh.shape)
        dbh = numpy.zeros(self.bh.shape)

        # calculate dl/dh for the last h
        # dl/dh = dl/dy * dy/dh
        dh = self.Why.T @ dy

        # backpropagation through time
        for t in reversed(range(n)):
            # an intermediate value: dl/dh * (1 - h^2)
            temp = ((1 - self.last_hs[t+1] ** 2) * dh)
            # dl/db = dl/dh * (1 - h^2)
            dbh += temp
            # dl/dwhh = dl/dh * (1 - h^2) * h_{t-1}
            dwhh += temp @ self.last_hs[t].T
            # dl/dwxh = dl/dh * (1 - h^2) * x
            dwxh += temp @ self.last_inputs[t].T
            # next dl/dh = dl/dh * (1 - h^2) * Whh
            dh = self.Whh.T @ temp
        
        # clip to prevent exploding gradients
        for d in [dwxh, dwhh, dwhy, dbh, dby]:
            numpy.clip(d, -1, 1, out=d)
    
        # update weights and biases using gradient descent
        self.Whh -= learn_rate * dwhh
        self.Wxh -= learn_rate * dwxh
        self.Why -= learn_rate * dwhy
        self.bh -= learn_rate * dbh
        self.by -= learn_rate * dby

if __name__ == '__main__':
    # Create the vocabulary
    vocab = list(set([w for text in pos_peg.train_data.keys() for w in text.split(' ')]))
    vocab_size = len(vocab)

    word_to_idx = { w: i for i, w in enumerate(vocab) }
    idx_to_word = { i: w for i, w in enumerate(vocab) }

    def create_inputs(text):
        inputs = []
        for w in text.split(' '):
            v = numpy.zeros((vocab_size, 1))
            v[word_to_idx[w]] = 1
            inputs.append(v)
        return inputs

    rnn = RNN(vocab_size, 2)

    def process(data, backprop=True):
        items = list(data.items())
        random.shuffle(items)

        loss = 0
        num_correct = 0

        for x, y in items:
            inputs = create_inputs(x)
            target = int(y)

            # forward
            out, _ = rnn.forward(inputs)
            probs = softmax(out)

            # calculate loss / accuracy
            loss -= numpy.log(probs[target])
            num_correct += int(numpy.argmax(probs) == target)

            if backprop:
                # build dl/dy
                dl_dy = probs
                dl_dy[target] -= 1

                # backward
                rnn.backprop(dl_dy)
        # Convert NumPy scalar to Python scalar.
        if isinstance(loss, numpy.ndarray):
            loss = loss.item()
        return loss / len(data), num_correct / len(data)

    losses = []
    accuracies = []
    # training loop
    for epoch in range(500):
        train_loss, train_acc = process(pos_peg.train_data)
        if (epoch + 1) % 20 == 0:
            losses.append(train_loss)
            accuracies.append(train_acc)
            print('Epoch %d' % (epoch + 1))
            print('Train: loss %.3f | accuracy: %.3f' % (train_loss, train_acc))

            test_loss, test_acc = process(pos_peg.test_data, backprop=False)
            print('Test: loss %.3f | accuracy: %.3f' % (test_loss, test_acc))
    
    (predict, _) = rnn.forward(create_inputs('i am good'))
    print('Predict text "i am good":', softmax(predict))

    pyplot.plot(numpy.array(range(len(losses))) * 50, losses, label='Loss')
    pyplot.plot(numpy.array(range(len(accuracies))) * 50, accuracies, label='Accuracy')
    pyplot.grid(True)
    pyplot.legend()
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()
