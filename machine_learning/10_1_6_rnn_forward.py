import numpy
from pos_peg_dataset import train_data

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
        # init
        h = numpy.zeros((self.Whh.shape[0], 1))
        # Perform each step of the RNN.
        for _, x in enumerate(inputs):
            h = numpy.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
        # Compute the output.
        y = self.Why @ h + self.by
        return y, h

if __name__ == '__main__':
    # Create the vocabulary
    vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
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
    inputs = create_inputs('i am very good')
    out, h = rnn.forward(inputs)
    porbs = softmax(out)
    print(porbs)
