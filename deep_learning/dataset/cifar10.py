import os
import pickle
from matplotlib import pyplot
import numpy

CLASS_NAME = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer", 
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

if __name__ == '__main__':
    base_url = '/Users/admin/Downloads/cifar-10-batches-py'
    batch_list = ['data_batch_1',
                  'data_batch_2',
                  'data_batch_3',
                  'data_batch_4',
                  'data_batch_5',
                  'test_batch']

    x_train, y_train = [], []
    for batch in batch_list:
        with open(os.path.join(base_url, batch), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            images = dict[b'data']
            labels = dict[b'labels']
            images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            x_train.append(images)
            y_train.append(labels)
    
    x_train = numpy.concatenate(x_train)
    y_train = numpy.concatenate(y_train)
    print(x_train.shape)
    print(y_train.shape)

    for i in range(9):
        pyplot.subplot(3, 3, i + 1)
        pyplot.imshow(x_train[i])
        pyplot.title(CLASS_NAME[y_train[i]])
        pyplot.axis('off')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.02)
    pyplot.show()
