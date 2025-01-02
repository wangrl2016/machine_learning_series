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

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.parse('./temp/mnist')
    conv_layer = Conv3x3(8)
    output = conv_layer.forward(train_images[0])
    assert output.shape == (26, 26, 8)
