import numpy
from deep_learning.dataset.pos_peg import train_data, test_data

# Create the vocabulary
vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unidque words found' % vocab_size)

# Assign indices to each word.
word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }
print(word_to_idx['good'])
print(idx_to_word[0])

def create_inputs(text):
    '''
    Return an array of one-hot vectors representing the words
    in the input text string.
    - text is a string
    - each ont-hot vector has shape (vocab_size, 1)
    '''
    inputs = []
    for w in text.split(' '):
        v = numpy.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    return inputs
