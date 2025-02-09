
from matplotlib import pyplot
import tensorflow as tf
import tensorflow_datasets as tfds

if __name__ == '__main__':
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print('Examples in Portuguese:')
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print('Examples in English:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))



