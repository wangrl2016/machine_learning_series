import unicodedata
import tensorflow as tf
import einops
import pathlib
import keras
import numpy
from matplotlib import pyplot

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
    return unicodedata.normalize(form, text).encode('utf-8') # type: ignore
    # def _normalize(s):
    #     s = s.numpy().decode('utf-8')  # Decode after getting NumPy value
    #     normalized_s = unicodedata.normalize(form, s) # type: ignore
    #     return tf.constant(normalized_s.encode('utf-8'))

    # # Handle both single strings and batches of strings
    # if isinstance(text, tf.Tensor) and text.shape == (): # single string
    #     return _normalize(text) # pass tf.Tensor to _normalize
    # else:  # Batch of strings (or already normalized Python string)
    #     return tf.map_fn(_normalize, text, dtype=tf.string)

def tf_lower_and_split_punct(text):
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

class Encoder(keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units
        
        # The embedding layer converts tokens to vectors.
        self.embedding = keras.layers.Embedding(self.vocab_size, units, mask_zero=True)
        
        # The RNN layers processes those vectors sequentially.
        self.rnn = keras.layers.Bidirectional(
            merge_mode='sum',
            layer= keras.layers.GRU(units,
                                    # Return the sequence and state
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform'))
        
    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch s')
        
        # 2. The embedding layer looks up the embedding vector for each token.
        x = self.embedding(x)
        shape_checker(x, 'batch s units')
        
        # 3. The GRU processes the sequence of embeddings.
        x = self.rnn(x)
        shape_checker(x, 'batch s units')
        
        # 4 Returns the new sequence of embeddings.
        return x
    
    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis] # type: ignore
        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context

class CrossAttention(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = keras.layers.MultiHeadAttention(key_dim=units, num_heads=1, **kwargs)
        self.layernorm = keras.layers.LayerNormalization()
        self.add = keras.layers.Add()

    def call(self, x, context):
        shape_checker = ShapeChecker()
        shape_checker(x, 'batch t units')
        shape_checker(context, 'batch s units')
        
        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)
        
        shape_checker(x, 'batch t units')
        shape_checker(attn_scores, 'batch heads t s')
        
        # Cache the attention scores for plotting later.
        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, 'batch t s')
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


if __name__ == '__main__':
    print(tf.__version__)
    path_to_zip = keras.utils.get_file(
        'spa-eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    
    # Dir 'spa-eng_extracted/' for some computer. 
    path_to_file = pathlib.Path(path_to_zip).parent/'spa-eng_extracted/spa-eng/spa.txt'
    
    target_raw, context_raw = load_data(path_to_file)
    print('Last context sentence:', context_raw[-1])
    print('Last target sentence:', target_raw[-1])

    example_text = '¿Todavía está en casa?' # Is he still at home?
    print(tf.constant(example_text).numpy())
    print(normalize_utf8(example_text, 'NFKD'))

    print(tf.constant(example_text).numpy().decode()) # type: ignore
    print(tf_lower_and_split_punct(normalize_utf8(example_text)).numpy().decode())

    context_raw = numpy.vectorize(normalize_utf8)(context_raw)
    # print('Last context sentence:', context_raw[-1])

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

    max_vocab_size = 5000
    
    context_text_processor = keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct, # type: ignore
        max_tokens=max_vocab_size,
        ragged=True)

    context_text_processor.adapt(train_raw.map(lambda context, target: context))
    # Here are the first 10 words from the vacabulary:
    print(context_text_processor.get_vocabulary()[:10])

    target_text_processor = keras.layers.TextVectorization(
        standardize=tf_lower_and_split_punct, # type: ignore
        max_tokens=max_vocab_size,
        ragged=True)
    target_text_processor.adapt(train_raw.map(lambda context, target: target))
    print(target_text_processor.get_vocabulary()[:10])

    example_tokens = context_text_processor(example_context_strings)
    print(example_tokens[:3, :])

    context_vocab = numpy.array(context_text_processor.get_vocabulary())
    tokens = context_vocab[example_tokens[0].numpy()]
    print(' '.join(tokens))

    pyplot.subplot(1, 2, 1)
    pyplot.pcolormesh(example_tokens.to_tensor())
    pyplot.title('Token IDs')
    pyplot.subplot(1, 2, 2)
    pyplot.pcolormesh(example_tokens.to_tensor() != 0)
    pyplot.title('Mask')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.06)
    pyplot.show()

    def process_text(context, target):
        # print('Context type:', type(context))
        # print('Context shape:', tf.shape(context))
        context = context_text_processor(context).to_tensor()
        target = target_text_processor(target)
        targ_in = target[:, :-1].to_tensor()
        targ_out = target[:, 1:].to_tensor()
        return (context, targ_in), targ_out
    
    train_ds = train_raw.map(process_text, tf.data.AUTOTUNE)
    val_ds = val_raw.map(process_text, tf.data.AUTOTUNE)

    for (ex_context_tok, ex_tar_in), ex_tar_out in train_ds.take(1): # type: ignore
        print(ex_context_tok[0, :10])
        tokens = numpy.array(context_text_processor.get_vocabulary())[ex_context_tok[0, :10].numpy()]
        print(' '.join(tokens))
        print(ex_tar_in[0, :10])
        tokens = numpy.array(target_text_processor.get_vocabulary())[ex_tar_in[0, :10].numpy()]
        print(' '.join(tokens))
        print(ex_tar_out[0, :10])
        tokens = numpy.array(target_text_processor.get_vocabulary())[ex_tar_out[0, :10].numpy()]
        print(' '.join(tokens))

    UNITS = 256
    # Encoder the input sequence.
    encoder = Encoder(context_text_processor, UNITS)
    ex_context = encoder(ex_context_tok)
    print(f'Context tokens, shape (batch, s): {ex_context_tok.shape}')
    print(f'Encoder output, shape (batch, s, units): {ex_context.shape}')

    attention_layer = CrossAttention(UNITS)
    # Attend to the encoded tokens.
    embed = keras.layers.Embedding(target_text_processor.vocabulary_size(),
                                   output_dim=UNITS, mask_zero=True)
    ex_tar_embed = embed(ex_tar_in)
    result = attention_layer(ex_tar_embed, ex_context)
    print(f'Context sequence, shape (batch, s, units): {ex_context.shape}')
    print(f'Target sequence, shape (batch, t, units): {ex_tar_embed.shape}')
    print(f'Attention result, shape (batch, t, units): {result.shape}')
    print(f'Attention weights, shape (batch, t, s): {attention_layer.last_attention_weights.shape}')

    print(attention_layer.last_attention_weights[0].numpy().sum(axis=-1))

    attention_weights = attention_layer.last_attention_weights
    mask = (ex_context_tok != 0).numpy() # type: ignore

    pyplot.subplot(1, 2, 1)
    pyplot.pcolormesh(mask*attention_weights[:, 0, :])
    pyplot.title('Attention weights')

    pyplot.subplot(1, 2, 2)
    pyplot.pcolormesh(mask)
    pyplot.title('Mask')
    pyplot.subplots_adjust(left=0.08, right=0.92, top=0.94, bottom=0.06)
    pyplot.show()
