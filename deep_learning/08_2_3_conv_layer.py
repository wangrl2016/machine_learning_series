import numpy

class Conv3x3:
    # A convolution layer using 3 x 3 filters.
    def __init__(self, num_filters):
        self.num_filters = num_filters
        # Filtersis a 3d array with dimensions (num_filters, 3, 3).
        # We divide by 9 to reduce the variance of our initial values.
        rng = numpy.random.default_rng()
        self.filters = rng.standard_normal(size=(num_filters, 3, 3)) / 9.0

    def iterate_regions(self, image):
        # Generates all possible 3 x 3 image regions using valid padding.
        # Param: image is a 2d numpy array.
        height, width = image.shape
        for i in range(height - 2):
            for j in range(width - 2):
                image_region = image[i: (i + 3), j: (j + 3)]
                yield image_region, i, j

    def forward(self, input):
        # Performs a forward pass of the conv layer using the given input.
        # Returns a 3d numpy array with dimension (h, w, num_filters).
        # Param: input is a 2d numpy array.
        height, width = input.shape
        output = numpy.zeros((height - 2, width - 2, self.num_filters))
        
        for image_region, i, j in self.iterate_regions(input):
            output[i, j] = numpy.sum(image_region * self.filters, axis=(1, 2))
        return output
