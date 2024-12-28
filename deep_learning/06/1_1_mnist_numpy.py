import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy
from dataset import mnist
from matplotlib import pyplot

def one_hot_encoding(labels, dimension=10):
    # Define a one-hot variable for an all-zeor vector
    # with 10 dimensions (number labels from 0 to 9).
    one_hot_labels = labels[..., None] == numpy.arange(dimension)[None]
    # Return one-hot encoded labels.
    return one_hot_labels.astype(numpy.float64)

# Define ReLU that returns the input if it's positive and 0 otherwise.
def relu(x):
    return (x >= 0) * x

# Set up a derivative of the ReLU function returns 1 for a positive input and 0 otherwise.
def relu2deriv(output):
    return output >= 0

learning_rate = 0.005
epochs = 20
hidden_size = 100
pixels_per_image = 784
num_labels = 10

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.parse()
    print('The shape of training images: {} and training labels: {}'.
          format(x_train.shape, y_train.shape))
    print('The shape of test iamges: {} and test labels: {}'.
          format(x_test.shape, y_test.shape))
    # Take the last image from the training set.
    last_image = x_train[-1, :]
    # Set the color mapping to grayscale to have a black background.
    pyplot.imshow(last_image, cmap='gray')
    pyplot.show()
    # Display the label of the last image from the training set.
    print('The last image label is', y_train[-1])

    print('The data type of training images: {}'.format(x_train.dtype))
    print('The data type of test images: {}'.format(x_test.dtype))

    x_train = x_train.reshape(-1, pixels_per_image)
    x_test = x_test.reshape(-1, pixels_per_image)
    train_sample, test_sample = 1000, 1000
    x_train = x_train[0:train_sample] / 255
    x_test = x_test[0:test_sample] / 255
    print('The data type of training images: {}'.format(x_train.dtype))
    print('The data type of test images: {}'.format(x_test.dtype))

    print('The data type of training labels: {}'.format(y_train.dtype))
    print('The data type of test labels: {}'.format(y_test.dtype))

    train_labels = one_hot_encoding(y_train[0:train_sample])
    test_labels = one_hot_encoding(y_test[0:test_sample])
    print('First one-hot is', train_labels[0])
    print('First train label is', y_train[0])
    
    rng = numpy.random.default_rng(0)
    weights_1 = 0.2 * rng.random((pixels_per_image, hidden_size)) - 0.1
    weights_2 = 0.2 * rng.random((hidden_size, num_labels)) - 0.1

    # To store training and test set losses and accurate predictions for visualization.
    store_train_loss = []
    store_train_accurate_pred = []
    store_test_loss = []
    store_test_accurate_pred = []

    # This is a training loop.
    # Run the learning experiment for a defined number of epochs (iterations).
    for j in range(epochs):
        # Set the initial loss/error and the number of accurate predictions to zero.
        training_loss = 0.0
        training_accurate_predictions = 0

        # For all images in the training set, perform a forward pass
        # and backpropagation and adjust the weights accordingly.
        for i in range(len(x_train)):
            # Forward propagation/forward pass:
            # 1. The input layer:
            #   Initialize the training image data as inputs.
            layer_0 = x_train[i]
            # 2. The hidden layer:
            #   Take in the training image data into the middle layer by
            #   matrix-multiplying it by randomly initialized weights.
            layer_1 = numpy.dot(layer_0, weights_1)
            # 3. Pass the hidden layer's output through the ReLu activation function.
            layer_1 = relu(layer_1)
            # 4. Define the dropout function for regularization.
            # dropout_mask = rng.integers(low=0, high=2, size=layer_1.shape)
            # 5. Apply dropout to the hidden layer's output.
            # layer_1 *= dropout_mask * 2
            # 6. The output layer:
            #   Ingest the output of the middle layer into the final layer
            #   by matrix-multiplying it by randomly initialized weights.
            #   Produce a 10-dimension vector with 10 scores.
            layer_2 = numpy.dot(layer_1, weights_2)

            # Backpropagation/backward pass:
            # 1. Measure the training error (loss function) between the actual
            #   image labels (the truth) and the prediction by the model.
            training_loss += numpy.sum((train_labels[i] - layer_2) ** 2)
            # 2. Increment the accurate prediction count.
            training_accurate_predictions += int(numpy.argmax(layer_2) == numpy.argmax(train_labels[i]))
            # 3. Differentiate the loss function/error.
            layer_2_delta = 2 * (train_labels[i] - layer_2)
            # 4. Propagate the gradients of the loss function back through the hidden layer.
            layer_1_delta = numpy.dot(weights_2, layer_2_delta) * relu2deriv(layer_1)
            # 5. Apply the dropout to the gradients.
            # layer_1_delta *= dropout_mask
            # 6. Update the weights for the middle and input layers
            #   by multiplying them by the learning rate and the gradients.
            weights_1 += learning_rate * numpy.outer(layer_0, layer_1_delta)
            weights_2 += learning_rate * numpy.outer(layer_1, layer_2_delta)

        # Store training set losses and accurate predictions.
        store_train_loss.append(training_loss)
        store_train_accurate_pred.append(training_accurate_predictions)

        # Evaluate model performance on the test set at each epoch.
        # Unlike the training step, the weights are not modified for each image
        # (or batch). Therefore the model can be applied to the test images in a
        # vectorized manner, eliminating the need to loop over each image
        # individually.
        results = relu(x_test @ weights_1) @ weights_2
        # Measure the error between the actual label (truth) and prediction values.
        test_loss = numpy.sum((test_labels - results) ** 2)
        # Measure prediction accuracy on test set.
        test_accurate_predictions = numpy.sum(
            numpy.argmax(results, axis=1) == numpy.argmax(test_labels, axis=1))
        # Store test set losses and accurate predictions.
        store_test_loss.append(test_loss)
        store_test_accurate_pred.append(test_accurate_predictions)

        # Summarize error and accuracy metrics at each epoch.
        print((
            f'Epoch: {j}\n'
            f'  Train set error: {training_loss / len(x_train):.3f}\n'
            f'  Train set accuracy: { training_accurate_predictions / len(x_train):.3f}\n'
            f'  Test set error: {test_loss / len(x_test):.3f}\n'
            f'  Test set accuracy: { test_accurate_predictions / len(x_test):.3f}'
        ))

    epoch_range = numpy.arange(epochs) + 1
    # The train set metrics.
    train_metrics = {
        'accuracy': numpy.asarray(store_train_accurate_pred) / len(x_train),
        'error': numpy.asarray(store_train_loss) / len(x_train),
    }
    # The test set metrics.
    test_metrics = {
        'accuracy': numpy.asarray(store_test_accurate_pred) / len(x_test),
        'error': numpy.asarray(store_test_loss) / len(x_test)
    }
    # Display the plots.
    fig, axes = pyplot.subplots(nrows=1, ncols=2)
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.92, bottom=0.08)
    for ax, metrics, title in zip(axes,(train_metrics, test_metrics),
                                    ('Train set', 'Test set')):
        for metric, values in metrics.items():
            ax.plot(epoch_range, values, label=metric.capitalize())
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
    pyplot.show()
