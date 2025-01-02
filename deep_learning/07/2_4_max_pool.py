import numpy
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import mnist

class Conv3x3:
    # A Convolution layer using 3x3 filters.
    def __init__(self, num_filters):
        self.num_filters = num_filters
        rng = numpy.random.default_rng(0)
        # Filters is a 3d array with dimensions (num_filters, 3, 3).
        # We divide by 9 to reduce the variance of our initial values.
        self.filters = rng.random((num_filters, 3, 3)) / 9
    
    def iterate_regions(self, image):
        # Generates all possible 3x3 image regions using valid padding.
        assert len(image.shape) == 2
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i+3),j:(j+3)]
                yield im_region, i, j
                
    def forward(self, input):
        # Performs a forward pass of the conv layer using the given input.
        # Returns a 3d numpy array with dimensions (h, w, num_filters).
        assert len(input.shape) == 2
        h, w = input.shape
        output = numpy.zeros((h-2, w-2, self.num_filters))
        
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = numpy.sum(im_region * self.filters, axis=(1, 2))
        return output

class MaxPool2:
    # A Max Pooling layer using a pool size of 2.
    def iterate_regions(self, image):
        # Generates non-overlapping 2x2 image regions to pool over.
        assert len(image.shape) == 3
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield im_region, i, j

    def forward(self, input):
        # Performs a forward pass of the maxpool layer using the given input.
        # Return a 3d numpy array with dimensions (h/2, w/2, num_filters).
        assert len(input.shape) == 3
        h, w, num_filters = input.shape
        output = numpy.zeros((h // 2, w // 2, num_filters))
        for im_region, i, j in self.iterate_regions(input):
            output[i, j] = numpy.amax(im_region, axis=(0, 1))
        return output

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.parse('./temp/mnist')
    conv_layer = Conv3x3(8)
    pool_layer = MaxPool2()
    
    output = conv_layer.forward(train_images[0])
    output = pool_layer.forward(output)
    assert (output.shape == (13, 13, 8))
