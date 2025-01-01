import numpy
from PIL import Image

if __name__ == '__main__':
    image = Image.open('res/deep_learning/flower.jpeg')
    img_arr = numpy.array(image)
    print('Dimension of image is',img_arr.shape)
    image.show()
