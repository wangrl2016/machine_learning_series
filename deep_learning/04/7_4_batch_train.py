import numpy
import sklearn
from matplotlib import pyplot

nn_input_dim = 2    # input layer dimensionality
nn_output_dim = 2   # output layer dimensionality
learn_rate = 0.01
reg_lambda = 0.01

def calculate_loss(model):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions.
    z1 = x.dot(W1) + b1
    a1 = numpy.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = numpy.exp(z2)
    probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss.
    corect_logprobs = -numpy.log(probs[range(num_examples), y])
    data_loss = numpy.sum(corect_logprobs)
    # Add regulatization term to loss (optional).
    data_loss += reg_lambda / 2 * (numpy.sum(numpy.square(W1)) + numpy.sum(numpy.square(W2)))
    return 1.0 / num_examples * data_loss

def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    z1 = x.dot(W1) + b1
    a1 = numpy.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = numpy.exp(z2)
    probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)
    return numpy.argmax(probs, axis=1)

def build_model(nn_hdim, epoches=1000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    rng = numpy.random.default_rng(0)
    W1 = rng.random((nn_input_dim, nn_hdim)) / numpy.sqrt(nn_input_dim)
    b1 = numpy.zeros((1, nn_hdim))
    W2 = rng.random((nn_hdim, nn_output_dim)) / numpy.sqrt(nn_hdim)
    b2 = numpy.zeros((1, nn_output_dim))

    # This is what we return at the end.
    model = {}

    # Gradient descent, for each batch...
    for i in range(0, epoches):
        for index, batch_x in enumerate(x_batches):
            z1 = batch_x.dot(W1) + b1
            a1 = numpy.tanh(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = numpy.exp(z2)
            probs = exp_scores / numpy.sum(exp_scores, axis=1, keepdims=True)

            # backpropagation
            delta3 = probs
            delta3[range(batch_size), y_batches[index]] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = numpy.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - numpy.power(a1, 2))
            dW1 = numpy.dot(batch_x.T, delta2)
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

        if print_loss and (i % 10 == 0):
            print('Loss after iteration %i: %f'  % (i, calculate_loss(model)))
    return model

def plot_decision_boundary(pred_func): 
    # Set min and max values and give it some padding 
    x_min, x_max = x[:, 0].min() - .5, x[:, 0].max() + .5 
    y_min, y_max = x[:, 1].min() - .5, x[:, 1].max() + .5 
    h = 0.01 
    # Generate a grid of points with distance h between them 
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, h),
                            numpy.arange(y_min, y_max, h)) 
    # Predict the function value for the whole gid 
    Z = pred_func(numpy.c_[xx.ravel(), yy.ravel()]) 
    Z = Z.reshape(xx.shape) 
    # Plot the contour and training examples 
    pyplot.contourf(xx, yy, Z, cmap='Wistia', alpha=0.8)
    pyplot.scatter(x[:, 0], x[:, 1], c=y)
    pyplot.grid(True)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.06)
    pyplot.show()

if __name__ == '__main__':
    x, y = sklearn.datasets.make_moons(32*6, noise=0.2)
    num_examples = len(x)
    indices = numpy.random.permutation(num_examples)
    x_shuffled = x[indices]
    y_shuffled = y[indices]
    batch_size = 16
    x_batches = [x_shuffled[i:i+batch_size] for i in range(0, len(x), batch_size)]
    y_batches = [y_shuffled[i:i+batch_size] for i in range(0, len(y), batch_size)]

    model = build_model(3, epoches=500, print_loss=True)
    plot_decision_boundary(lambda x: predict(model, x))