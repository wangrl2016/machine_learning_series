import numpy
import operator
import nltk
import csv
import itertools
import time

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

    def bptt(self, x, y):
        T = len(y)
        # Perform forward propagation.
        (o, s) = self.forward_propagation(x)
        # We accumulate the gradients in these variables.
        dldU = numpy.zeros(self.U.shape)
        dldV = numpy.zeros(self.V.shape)
        dldW = numpy.zeros(self.W.shape)
        delta_o = o
        delta_o[numpy.arange(len(y)), y] -= 1.0
        # For each output backwards.
        for t in numpy.arange(T)[::-1]:
            dldV += numpy.outer(delta_o[t], s[t].T)
            # Initial delta calculation.
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t]**2))
            # Backpropagation through time (for at most self.bptt_trancate steps)
            for bptt_step in numpy.arange(max(0, t - self.bptt_truncate), t+1)[::-1]:
                dldW += numpy.outer(delta_t, s[bptt_step-1])
                dldU[:, x[bptt_step]] += delta_t
                # Update delta for next step.
                delta_t = self.W.T.dot(delta_t) * (1 - s[bptt_step-1] ** 2)
        return (dldU, dldV, dldW)

    def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
        # Calculate the gradients using backpropagation. We want to checker if these are correct.
        bptt_gradients = self.bptt(x, y)
        # List of all parameters we want to check.
        model_parameters = ['U', 'V', 'W']
        # Gradient check for each parameter
        for pidx, pname in enumerate(model_parameters):
            # Get the actual parameter value from the mode, e.g. model.W
            parameter = operator.attrgetter(pname)(self)
            # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
            it = numpy.nditer(parameter, flags=['multi_index'])
            while not it.finished:
                ix = it.multi_index
                # Save the original value so we can reset it later
                original_value = parameter[ix]
                # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
                parameter[ix] = original_value + h
                gradplus = self.calculate_total_loss([x],[y])
                parameter[ix] = original_value - h
                gradminus = self.calculate_total_loss([x],[y])
                estimated_gradient = (gradplus - gradminus)/(2*h)
                # Reset parameter to original value
                parameter[ix] = original_value
                 # The gradient for this parameter calculated using backpropagation
                backprop_gradient = bptt_gradients[pidx][ix]
                # calculate The relative error: (|x - y|/(|x| + |y|))
                relative_error = numpy.abs(backprop_gradient - estimated_gradient) 
                # If the error is to large fail the gradient check.
                if relative_error > error_threshold * (numpy.abs(backprop_gradient) + numpy.abs(estimated_gradient)):
                    print('Gradient check error: parameter = ' + pname + ', ix = ' + str(ix))
                    print('+h loss:', gradplus)
                    print('-h loss:', gradplus)
                    print('Estimated_gradient:', estimated_gradient)
                    print('Backpropagation gradient:', backprop_gradient)
                    print('Relative error:', relative_error)
                    return
                it.iternext()
            print('Gradient check for parameter ' + pname + ' passed')

    def sgd_step(self, x, y, learning_rate):
        # Calculate the gradients
        dldU, dldV, dldW = self.bptt(x, y)
         # Change parameters according to gradients and learning rate
        self.U -= learning_rate * dldU
        self.V -= learning_rate * dldV
        self.W -= learning_rate * dldW

def train_with_sgd(model, x_train, y_train, learning_rate=0.005, epochs=100, evaluate_loss_after=5):
    losses = []
    num_examples_seen = 0
    for epoch in range(epochs):
        # Optionally evaluate the loss
        if epoch % evaluate_loss_after == 0:
            loss = model.calculate_loss(x_train, y_train)
            losses.append((num_examples_seen, loss))
            print('Loss after num_example_seen=' + str(num_examples_seen) + ' epoch=' + str(epoch) + ' - ' + str(loss))
            # Adjust the learning rate if loss increases.
            if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                learning_rate = learning_rate * 0.5
        # For each training example...
        for i in range(len(y_train)):
            model.sgd_step(x_train[i], y_train[i], learning_rate)
            num_examples_seen += 1

def generate_sentence(model):
    # We start the sentence with the start token.
    new_sentence = [int(word_to_index[sentence_start_token])]
    # Repeat until we get an end token.
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        (next_word_probs, _) = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words.
        while sampled_word == word_to_index[unknown_token]:
            samples = numpy.random.multinomial(1, next_word_probs[-1])
            sampled_word = int(numpy.argmax(samples))
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

vocabulary_size = 8000
unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'

if __name__ == '__main__':
    nltk.download('punkt_tab')
    # Read the data and append SENTENCE_START and SENTENCE_END tokens.
    with open('temp/reddit-comments-2015-08.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences.
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END.
        sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token) for x in sentences]

    # Tokenize the sentences into words.
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Count the word frequencies.
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))

    # Get the most common words and build index_to_word and word_to_index vectors.
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])

    # Replace all words in our vocabulary with the unkown token.
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    # Create the training data.
    x_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences], dtype=object)
    y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences], dtype=object)

    model = RNNNumpy(vocabulary_size)
    start = time.time()
    model.sgd_step(x_train[10], y_train[10], 0.005)
    duration = time.time() - start
    print('Train sample 10 duration:', duration)

    # Train on a small subset of the data to see what happen
    model = RNNNumpy(vocabulary_size)
    losses = train_with_sgd(model, x_train[:100], y_train[:100], epochs=10, evaluate_loss_after=1)

    # Train on a middle subset of the data
    losses = train_with_sgd(model, x_train[:5000], y_train[:5000], epochs=10, evaluate_loss_after=1)

    # predict words
    num_sentences = 10
    sentence_min_length = 7
    for i in range(num_sentences):
        sent = []
        while len(sent) < sentence_min_length:
            sent = generate_sentence(model)
        print(' '.join(sent))
