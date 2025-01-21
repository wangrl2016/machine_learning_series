import numpy
import operator

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

if __name__ == '__main__':
    grad_check_vocab_size = 100
    model = RNNNumpy(grad_check_vocab_size, 10, bptt_truncate=1000)
    model.gradient_check([1, 2, 3, 5], [2, 3, 4, 5])
