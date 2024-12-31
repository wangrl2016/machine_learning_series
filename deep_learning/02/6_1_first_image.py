import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import mnist
from matplotlib import pyplot
import random

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.parse()
    print('First image label is', y_train[0])
    first_image = x_train[0]
    first_image[first_image < 50] += 50
    print(first_image)
    pyplot.imshow(first_image, cmap='gray', vmin=0, vmax=255)
    pyplot.axis('off')
    pyplot.savefig('temp/mnist_first_image.png',
                   bbox_inches='tight', pad_inches=0)
    pyplot.show()
