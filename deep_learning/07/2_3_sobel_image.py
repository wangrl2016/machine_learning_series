import numpy
from PIL import Image
import scipy
import scipy.ndimage
from matplotlib import pyplot

if __name__ == '__main__':
    image = Image.open('res/deep_learning/lena.jpg')
    img_arr = numpy.array(image)

    sobel = numpy.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
    sobel_3d = numpy.repeat(sobel[numpy.newaxis, :, :], 3, axis=0)
    gradient = scipy.ndimage.convolve(img_arr, sobel_3d)
    pyplot.figure(figsize=(6, 3))
    pyplot.subplot(1, 2, 1)
    pyplot.imshow(image)
    pyplot.axis('off')
    pyplot.subplot(1, 2, 2)
    pyplot.imshow(gradient)
    pyplot.axis('off')
    pyplot.tight_layout()
    pyplot.show()
