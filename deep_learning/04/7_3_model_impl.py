import numpy
import sklearn

nn_input_dim = 2    # input layer dimensionality
nn_output_dim = 2   # output layer dimensionality
learn_rate = 0.01
reg_lambda = 0.01

def calculate_loss(model):
    return 0.0

def build_model(nn_hdim, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    rng = numpy.random.default_rng(0)
    W1 = rng.random((nn_input_dim, nn_hdim)) / numpy.sqrt(nn_input_dim)
    b1 = numpy.zeros((1, nn_hdim))
    W2 = rng.random((nn_hdim, nn_output_dim)) / numpy.sqrt(nn_hdim)
    b2 = numpy.zeros((1, nn_output_dim))

    # This is what we return at the end.
    model = {}

    # Gradient descent, for each batch...
    for i in range(0, num_passes):
        z1 = x.dot(W1) + b1
        a1 = numpy.tanh(z1)
        z2 = a1.dot(W2) + b2
        exp_scores = numpy.exp(z2)
        probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)

        # backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = numpy.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - numpy.power(a1, 2))
        dW1 = numpy.dot(x.T, delta2)
        db1 = numpy.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms).
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        W1 += -learn_rate * dW1
        b1 += -learn_rate * db1
        W2 += -learn_rate * dW2
        b2 += -learn_rate * db2

        # Assign new prameters to the model.
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        if print_loss and i % 1000 == 0:
            print('Loss after iteration %i: %f'  % (i, calculate_loss(model)))
    return model 

if __name__ == '__main__':
    x, y = sklearn.datasets.make_moons(200, noise=0.2)
    num_examples = len(x)

    model = build_model(3, print_loss=True)
