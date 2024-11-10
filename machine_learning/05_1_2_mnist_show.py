import mnist_parse
import os
from matplotlib import pyplot
import random

if __name__ == '__main__':
    # base_url = '/Users/admin/Downloads/'
    # filename_list = ['train-images-idx3-ubyte',
    #                   'train-labels-idx1-ubyte',
    #                   't10k-images-idx3-ubyte',
    #                   't10k-labels-idx1-ubyte']
    (x_train, y_train), (x_test, y_test) = mnist_parse.dataset()

    images_show = []
    labels_show = []
    row = 5
    col = 3
    random.seed(100)
    for i in range(0, row * 2):
        r = random.randint(0, len(x_train))
        images_show.append(x_train[r])
        labels_show.append(str(r) + ' ' + str(y_train[r]))
    for i in range(0, row):
        r = random.randint(0, len(x_test))
        images_show.append(x_test[r])
        labels_show.append(str(r) + ' ' + str(y_test[r]))
    index = 1
    for image, label in zip(images_show, labels_show):
        ax = pyplot.subplot(col, row, index)
        ax.set_xticks([])  # remove x-axis ticks
        ax.set_yticks([])
        pyplot.imshow(image, cmap='gray')
        pyplot.title(label)
        index += 1
    pyplot.tight_layout()
    pyplot.subplots_adjust(left=0.04, right=0.96, top=0.96, bottom=0.01)
    pyplot.savefig('temp/mnist_dataset_sample.png')
    pyplot.show()
