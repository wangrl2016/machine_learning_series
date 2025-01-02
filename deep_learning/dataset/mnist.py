import array
import numpy
import os
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

# def parse():
#     base_url = 'https://yann.lecun.com/exdb/mnist/'
#     filename_list = ['train-images-idx3-ubyte',
#                       'train-labels-idx1-ubyte',
#                       't10k-images-idx3-ubyte',
#                       't10k-labels-idx1-ubyte']
#     for filename in filename_list:
#         url = os.path.join(base_url, filename) + '.gz'
#         file_util.get_file(origin_url=url, dest_file_name=filename)

# Load the MNIST dataset.
def mnist_load(train_image_path, train_label_path,
               test_image_path, test_label_path):
    return mnist_read(train_image_path, train_label_path), \
        mnist_read(test_image_path, test_label_path)

def parse(base_url='/Users/admin/Downloads/'):
    filename_list = ['train-images-idx3-ubyte',
                      'train-labels-idx1-ubyte',
                      't10k-images-idx3-ubyte',
                      't10k-labels-idx1-ubyte']
    return mnist_load(*[os.path.join(base_url, filename) for filename in filename_list])
