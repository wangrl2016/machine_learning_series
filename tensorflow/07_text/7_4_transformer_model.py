
from matplotlib import pyplot
import tensorflow as tf
import tensorflow_text as tf_text
import tensorflow_datasets as tfds
import keras
import numpy

if __name__ == '__main__':
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                   with_info=True,
                                   as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']

    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print('> Examples in Portuguese:')
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print('> Examples in English:')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    model_name = 'ted_hrlr_translate_pt_en_converter'
    keras.utils.get_file(
        f'{model_name}.zip',
        f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
        cache_dir='.', cache_subdir='', extract=True)
    tokenizers = tf.saved_model.load('ted_hrlr_translate_pt_en_converter_extracted/' + model_name)

    print('> This is a batch of strings:')
    for en in en_examples.numpy():
        print(en.decode('utf-8'))

    encoded = tokenizers.en.tokenize(en_examples)
    print('> This is a padded-batch of token IDs:')
    for row in encoded.to_list():
        print(row)

    print('> This is the text split into tokens:')
    tokens = tokenizers.en.lookup(encoded)
    print(tokens)

    lengths = []
    for pt_examples, en_examples in train_examples.batch(1024):
        pt_tokens = tokenizers.pt.tokenize(pt_examples)
        lengths.append(pt_tokens.row_lengths())
    
        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(en_tokens.row_lengths())
        print('.', end='', flush=True)
    print()

    all_lengths = numpy.concatenate(lengths)
    pyplot.hist(all_lengths, numpy.linspace(0, 500, 101))
    pyplot.ylim(pyplot.ylim())
    max_length = max(all_lengths)
    pyplot.plot([max_length, max_length], pyplot.ylim())
    pyplot.title(f'Maximum tokens per examples: {max_length}')
    pyplot.show()

    MAX_TOKENS = 128
    def prepare_batch(pt, en):
        pt = tokenizers.pt.tokenize(pt)
        pt = pt[:, :MAX_TOKENS]    # trim to MAX_TOKENS
        pt = pt.to_tensor() # convert to 0-padded dense tensor

        en = tokenizers.en.tokenize(en)
        en = en[:, :(MAX_TOKENS + 1)]
        en_inputs = en[:, :-1].to_tensor() # drop the [END] tokens
        en_labels = en[:, 1:].to_tensor() # drop the [START] tokens

        return (en, en_inputs), en_labels

    BUFFER_SIZE = 20000
    BATCH_SIZE = 64

    def make_batches(ds):
        return (ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).map(
            prepare_batch, tf.data.AUTOTUNE).prefetch(buffer_size=tf.data.AUTOTUNE))
    
    # Create training and validation set batches.
    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    for (pt, en), en_labels in train_batches.take(1):
        break

    print(pt.shape)
    print(en.shape)
    print(en_labels.shape)
    print(en[0][:10])
    print(en_labels[0][:10])
