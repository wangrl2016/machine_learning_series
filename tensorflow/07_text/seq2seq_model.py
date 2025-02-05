import unicodedata
import tensorflow as tf
import einops
import pathlib
import keras
import numpy

class ShapeChecker():
    def __init__(self):
        # Keep a cache of every axis-name seen.
        self.shapes = {}
    
    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return
        
        parsed = einops.parse_shape(tensor, names)
        
        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if (broadcast and new_dim == 1):
                continue
            
            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue
            
            if new_dim != old_dim:
                raise ValueError(f'Shape mismatch for dimension: {name}\n'
                                 f'    found: {new_dim}\n'
                                 f'    expected: {old_dim}')

def load_data(path):
    text = path.read_text(encoding='utf-8')
    
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]
    
    context = numpy.array([context for target, context in pairs])
    target = numpy.array([target for target, context in pairs])
    
    return target, context

def normalize_utf8(text, form='NFKD'):
    def _normalize(s):
        s = s.numpy().decode('utf-8')  # Decode after getting NumPy value
        normalized_s = unicodedata.normalize(form, s) # type: ignore
        return tf.constant(normalized_s.encode('utf-8'))

    # Handle both single strings and batches of strings
    if isinstance(text, tf.Tensor) and text.shape == (): # single string
        return _normalize(text) # pass tf.Tensor to _normalize
    else:  # Batch of strings (or already normalized Python string)
        return tf.map_fn(_normalize, text, dtype=tf.string)

def tf_lower_and_split_punct(text):
    # Split accented characters.
    text = normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

if __name__ == '__main__':
    print(tf.__version__)
    path_to_zip = keras.utils.get_file(
        'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    
    # Dir 'spa-eng_extracted/' for some computer. 
    path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng_extracted/spa-eng/spa.txt'
    
    target_raw, context_raw = load_data(path_to_file)
    print('Train dataset len:', context_raw[-1])
    print(target_raw[-1])
    
    BUFFER_SIZE = len(context_raw)
    print(BUFFER_SIZE)
    BATCH_SIZE = 64
    
    is_train = numpy.random.uniform(size=(len(target_raw), )) < 0.8
    
    train_raw = (
        tf.data.Dataset.from_tensor_slices((context_raw[is_train], target_raw[is_train]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True))
    val_raw = (
        tf.data.Dataset.from_tensor_slices((context_raw[~is_train], target_raw[~is_train]))
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE, drop_remainder=True))
    
    for example_context_strings, example_target_strings in train_raw.take(1): # type: ignore
        print(example_context_strings[:5])
        print(example_target_strings[:5])
        break

    example_text = tf.constant('¿Todavía está en casa?') # Is he still at home?
    print(example_text.numpy()) # type: ignore
    print(normalize_utf8(example_text, 'NFKD').numpy()) # type: ignore
    
    print(example_text.numpy().decode()) # type: ignore
    print(tf_lower_and_split_punct(example_text).numpy().decode())
    
    max_vocab_size = 5000
    
    context_text_processor = keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct, # type: ignore
        max_tokens=max_vocab_size,
        ragged=True)

    context_text_processor.adapt(train_raw.map(lambda context, target: context))
    # Here are the first 10 words from the vacabulary:
    print(context_text_processor.get_vocabulary()[:10])
