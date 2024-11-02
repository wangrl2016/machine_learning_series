import array
import numpy
import struct

def mnist_read(images_path, labels_path):
    labels = []
    with open(labels_path, 'rb') as file:
        magic, size = struct.unpack('>II', file.read(8))
        if magic != 2049:
            raise ValueError('Magic number mismatch, got {}'.format(magic))
        labels = array.array('B', file.read())

    with open(images_path, 'rb') as file:
        magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
        if magic != 2051:
            raise ValueError('Magic number mismatch, got {}'.format(magic))
        image_data = array.array('B', file.read())

    images = []
    for k in range(size):
        images.append([0] * rows * cols)
    for j in range(size):
        img = numpy.array(image_data[j * rows * cols:(j + 1) * rows * cols])
        img = img.reshape(28, 28)
        images[j][:] = img

    return numpy.array(images), numpy.array(labels)

# Load the MNIST dataset.
def mnist_load(train_image_path, train_label_path,
               test_image_path, test_label_path):
    return mnist_read(train_image_path, train_label_path), \
        mnist_read(test_image_path, test_label_path)
