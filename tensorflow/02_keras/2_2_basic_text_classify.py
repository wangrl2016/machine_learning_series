import keras
from matplotlib import pyplot
import numpy
import tensorflow as tf
import os
import shutil

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

if __name__ == '__main__':
    print('TensorFlow version:', tf.__version__)

    dataset = keras.utils.get_file('aclImdb_v1', url,
                                   untar=True,
                                   cache_subdir='')
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb_v1/aclImdb')
    print(os.listdir(dataset_dir))
    train_dir = os.path.join(dataset_dir, 'train')
    test_dir = os.path.join(dataset_dir, 'test')
    print(os.listdir(train_dir))
    sample_file = os.path.join(train_dir, 'pos/1181_9.txt')
    with open(sample_file) as f:
        print(f.read())

    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)

    BATCH_SIZE = 32
    seed = 42
    raw_train_ds = keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='training',
        seed=seed)

    for text_batch, label_batch in raw_train_ds.take(1):
        for i in range(3):
            print('Review:', text_batch.numpy()[i])
            print('Label:', label_batch.numpy()[i])

    raw_val_ds = keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=seed)
