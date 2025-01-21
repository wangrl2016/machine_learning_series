import numpy
import csv
import itertools
import nltk

def softmax(x):
    xt = numpy.exp(x - numpy.max(x))
    return xt / numpy.sum(xt)

class RNNNumpy:
    def __init__(self, word_dim, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables.
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        # Randomly initialize the network parameters.
        rng = numpy.random.default_rng(0)
        self.U = rng.uniform(-numpy.sqrt(1./word_dim), numpy.sqrt(1./word_dim),
                             (hidden_dim, word_dim))
        self.V = rng.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim),
                             (word_dim, hidden_dim))
        self.W = rng.uniform(-numpy.sqrt(1./hidden_dim), numpy.sqrt(1./hidden_dim),
                             (hidden_dim, hidden_dim))

    def forward_propagation(self, x):
        # The total number of time steps.
        T = len(x)
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0.
        s = numpy.zeros((T + 1, self.hidden_dim))
        # The outputs at each time step. Again, we save them for later.
        o = numpy.zeros((T, self.word_dim))

        # For each time step...
        for t in numpy.arange(T):
            # Note that we are indexing U by x[t].
            # This is the same as multiplying U with a one-hot vector.
            s[t] = numpy.tanh(self.U[:, x[t]] + self.W.dot(s[t-1]))
            o[t] = softmax(self.V.dot(s[t]))
        return (o, s)

    def predict(self, x):
        # Perform forward propagation and return index of the highest score.
        o, s = self.forward_propagation(x)
        return numpy.argmax(o, axis=1)
    
    # 同时传递多个句子
    def calculate_total_loss(self, x, y):
        loss = 0
        # For each sentence...
        for i in numpy.arange(len(y)):
            (o, s) = self.forward_propagation(x[i])
            # We only care about our prediction of the 'correct' words.
            correct_word_predictions = o[numpy.arange(len(y[i])), y[i]]
            # Add to the loss based on how off we are.
            loss += -1 * numpy.sum(numpy.log(correct_word_predictions))
        return loss
    
    def calculate_loss(self, x, y):
        # Divide the total loss by the number of training examples.
        N = 0
        for y_i in y:
            N += len(y_i)
        return self.calculate_total_loss(x, y) / N
            

vocabulary_size = 8000
unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'

if __name__ == '__main__':
    nltk.download('punkt_tab')
    # Read the data and append SENTENCE_START and SENTENCE_END tokens.
    print('Reading CSV files...')
    with open('temp/reddit-comments-2015-08.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences.
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END.
        sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print('Parsed', len(sentences), 'sentences')
    
    # Tokenize the sentences into words.
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Count the word frequencies.
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print('Found', len(word_freq.items()), 'unique words tokens')

    # Get the most common words and build index_to_word and word_to_index vectors.
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    print('Using vocabulary size', vocabulary_size)
    print('The least frequent word in our vocabulary is', vocab[-1][0], 'and appeared', vocab[-1][1], 'times')

    # Replace all words in our vocabulary with the unkown token.
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    
    print('Example sentence 10:', sentences[10])
    print('Example sentence 10 after pre-processing:', tokenized_sentences[10])
    
    # Create the training data.
    x_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences], dtype=object)
    y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences], dtype=object)
    print('Sentence 10 source:', x_train[10])
    print('Sentence 10 target:', y_train[10])

    model = RNNNumpy(vocabulary_size)
    (o, s) = model.forward_propagation(x_train[10])
    print('Output shape:', o.shape)

    predictions = model.predict(x_train[10])
    print('Prediction shape:', predictions.shape)
    print('Prediction', predictions)

    print('Expect loss:', numpy.log(vocabulary_size))
    loss = model.calculate_loss(x_train[:1000], y_train[:1000])
    print('Actual loss:', loss)
