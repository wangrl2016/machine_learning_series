import keras
from matplotlib import pyplot
import numpy
import tensorflow as tf
import os
import shutil
import string
import re

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
    
    print('Label 0 corresponds to', raw_train_ds.class_names[0])
    print('Label 1 corresponds to', raw_train_ds.class_names[1])

    raw_val_ds = keras.utils.text_dataset_from_directory(
        train_dir,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=seed)

    raw_test_ds = keras.utils.text_dataset_from_directory(
        test_dir,
        batch_size=BATCH_SIZE)

    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        return tf.strings.regex_replace(stripped_html,
                                        '[%s]' % re.escape(string.punctuation),
                                        '')
    max_features = 10000
    sequence_length = 250
    vectorize_layer = keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    # Make a text-only dataset (without labels), then call adapt.
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label
    
    text_batch, label_batch = next(iter(raw_train_ds))
    first_review, first_label = text_batch[0], label_batch[0]
    print('First review:', first_review)
    print('First label:', raw_train_ds.class_names[first_label])
    print('First vectorized review:', vectorize_text(first_review, first_label))

    print('1287 --->', vectorize_layer.get_vocabulary()[1287])
    print('313 --->', vectorize_layer.get_vocabulary()[313])
    print('Vocabulary size:', str(len(vectorize_layer.get_vocabulary())))

    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)

    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    embedding_dim = 32

    model = keras.Sequential([
        keras.layers.Embedding(max_features, embedding_dim, trainable=True),
        keras.layers.Dropout(0.2),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()

    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer='adam',
                  metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)])

    EPOCHS = 10
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
    
    loss, accuracy = model.evaluate(test_ds)
    print('Loss:', loss)
    print('Accuracy:', accuracy)

    history_dict = history.history
    print(history_dict.keys())

    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, EPOCHS+1)
    pyplot.plot(epochs, loss, 'bo', label='Training loss')
    pyplot.plot(epochs, val_loss, 'b', label='Validation loss')
    pyplot.title("Training and validation loss")
    pyplot.legend()
    pyplot.grid()
    pyplot.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06)
    pyplot.show()

    pyplot.plot(epochs, acc, 'bo', label='Training acc')
    pyplot.plot(epochs, val_acc, 'b', label='Validation acc')
    pyplot.title('Training and validation accuracy')
    pyplot.legend()
    pyplot.grid()
    pyplot.subplots_adjust(left=0.08, right=0.98, top=0.94, bottom=0.06)
    pyplot.show()

    examples = tf.constant([
        "The movie was great!",
        "The movie was okay.",
        "The movie was terrible..."
    ])
    print(model.predict(vectorize_layer(examples)).squeeze())
    print(numpy.round(model.predict(vectorize_layer(text_batch)), 3).squeeze())
    print(label_batch)
