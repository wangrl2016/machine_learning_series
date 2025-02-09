import unicodedata
import tensorflow as tf
import einops
import pathlib
import keras
import numpy
from matplotlib import pyplot
from matplotlib import ticker

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


class Decoder(keras.layers.Layer):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun
    
    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]')
        self.id_to_word = keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token='', oov_token='[UNK]',
            invert=True)
        self.start_token = self.word_to_id('[START]')
        self.end_token = self.word_to_id('[END]')
        
        self.units = units
        
        # 1. The embedding layer converts token IDs to vectors.
        self.embedding = keras.layers.Embedding(self.vocab_size,
                                                units,
                                                mask_zero=True)
        # 2. The RNN keeps track of what's been generated so far.
        self.rnn = keras.layers.GRU(units,
                                    return_sequences=True,
                                    return_state=True,
                                    recurrent_initializer='glorot_uniform')
        # 3. The RNN output will be the query for the attention layer.
        self.attention = CrossAttention(units)
        
        # 4. This fully connected layer produces the logits for each output token.
        self.output_layer = keras.layers.Dense(self.vocab_size)

@Decoder.add_method
def call(self, context, x, state=None, return_state=False):
    shape_checker = ShapeChecker()
    shape_checker(x, 'batch t')
    shape_checker(context, 'batch s units')
    
    # 1. Lookup the embeddings
    x = self.embedding(x)
    shape_checker(x, 'batch t units')
    
    # 2. Process the target sequence.
    x, state = self.rnn(x, initial_state=state)
    shape_checker(x, 'batch t units')
    
    # 3. Use the RNN output as the query for the attention over the context.
    x = self.attention(x, context)
    self.last_attention_weights = self.attention.last_attention_weights
    shape_checker(x, 'batch t units')
    shape_checker(self.last_attention_weights, 'batch t s')
    
    # 4. Generate logit predictions for the next token.
    logits = self.output_layer(x)
    shape_checker(logits, 'batch t target_vocab_size')
    
    if return_state:
        return logits, state
    else:
        return logits

@Decoder.add_method
def get_initial_state(self, context):
    batch_size = tf.shape(context)[0]
    start_tokens = tf.fill([batch_size, 1], self.start_token)
    print(tf.shape(start_tokens))
    done = tf.zeros([batch_size, 1], dtype=tf.bool)
    return start_tokens, done
 
@Decoder.add_method
def tokens_to_text(self, tokens):
    words = self.id_to_word(tokens)
    result = tf.strings.reduce_join(words, axis=-1, separator=' ')
    result = tf.strings.regex_replace(result, r'^\s*\[START\]\s*', '')
    result = tf.strings.regex_replace(result, r'\s*\[END\]\s*$', '')
    return result

@Decoder.add_method
def get_next_token(self, context, next_token, done, temperature = 0.0):
    logits, state = self(
        context, next_token,
        return_state=True) 

    if temperature == 0.0:
        next_token = tf.argmax(logits, axis=-1)
    else:
        logits = logits[:, -1, :]/temperature
        next_token = tf.random.categorical(logits, num_samples=1)

    # If a sequence produces an `end_token`, set it `done`
    done = done | (next_token == self.end_token)
    # Once a sequence is done it only produces 0-padding.
    next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

    return next_token, done

class Translator(keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun
    
    def __init__(self, units,
               context_text_processor,
               target_text_processor):
        super().__init__()
        # Build the encoder and decoder
        encoder = Encoder(context_text_processor, units)
        decoder = Decoder(target_text_processor, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)
        return logits
    
@Translator.add_method
def translate(self, texts, *, max_length=50, temperature=0.0):
    # Process the input texts.
    context = self.encoder.convert_input(texts)
    batch_size = tf.shape(texts)[0]
    
    # Setup the loop inputs.
    tokens = []
    attention_weights = []
    next_token, done = self.decoder.get_initial_state(context)
    
    for _ in range(max_length):
        # Generate the next token.
        next_token, done = self.decoder.get_next_token(
            context, next_token, done, temperature)
        
        # Collect the generated tokens.
        tokens.append(next_token)
        attention_weights.append(self.decoder.last_attention_weights)
        
        if tf.executing_eagerly() and tf.reduce_all(done):
            break
        
    # Stack the lists of tokens and attention weights.
    tokens = tf.concat(tokens, axis=-1) # t*[(batch 1)] -> (batch, t)
    self.last_attention_weights = tf.concat(attention_weights, axis=1)  # t*[(batch 1 s)] -> (batch, t s)
    
    result = self.decoder.tokens_to_text(tokens)
    return result

@Translator.add_method
def plot_attention(self, text, **kwargs):
    assert isinstance(text, str)
    output = self.translate([text], **kwargs)
    output = output[0].numpy().decode()
    
    attention = self.last_attention_weights[0]
    context = tf_lower_and_split_punct(normalize_utf8(text))
    context = context.numpy().decode().split()
    
    output = tf_lower_and_split_punct(output)
    output = output.numpy().decode().split()[1:]
    
    fig = pyplot.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis', vmin=0.0)
    
    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + context, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + output, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    ax.set_xlabel('Input text')
    ax.set_ylabel('Output text')
    pyplot.show()

def masked_loss(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')
    loss = loss_fn(y_true, y_pred)
    
    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, loss.dtype) # type: ignore
    loss *= mask # type: ignore
    
    # Return the total
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
def masked_acc(y_true, y_pred):
    # Calculate the loss for each item in the batch.
    y_pred = tf.argmax(y_pred, axis=-1)
    y_pred = tf.cast(y_pred, y_true.dtype)

    match = tf.cast(y_true == y_pred, tf.float32)
    mask = tf.cast(y_true != 0, tf.float32)

    return tf.reduce_sum(match)/tf.reduce_sum(mask)

class Export(tf.Module):
    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(dtype=tf.string, shape=[None])])
    def translate(self, inputs):
        return self.model.translate(inputs)

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
    
    decoder = Decoder(target_text_processor, UNITS)
    logits = decoder(ex_context, ex_tar_in)
    print(f'encoder output shape: (batch, s, units) {ex_context.shape}')
    print(f'input target tokens shape: (batch, t) {ex_tar_in.shape}')
    print(f'logits shape shape: (batch, target_vocabulary_size) {logits.shape}')

    # Setup the loop variables.
    next_token, done = decoder.get_initial_state(ex_context)
    tokens = []
    
    for n in range(10):
        # Run one step.
        next_token, done = decoder.get_next_token(
            ex_context, next_token, done, temperature=1.0)
        # Add the token to the output.
        tokens.append(next_token)
    
    # Stack all the token together.
    tokens = tf.concat(tokens, axis=-1) # (batch, t)
    
    # Convert the tokens back to a string.
    result = decoder.tokens_to_text(tokens)
    print(result[:3].numpy())

    model = Translator(UNITS, context_text_processor, target_text_processor)
    
    logits = model((ex_context_tok, ex_tar_in))
    print(f'Context tokens, shape: (batch, s, units) {ex_context_tok.shape}')
    print(f'Target tokens, shape: (batch, t) {ex_tar_in.shape}')
    print(f'logits, shape: (batch, t, target_vocabulary_size) {logits.shape}')
    
    model.compile(optimizer='adam',
                  loss=masked_loss,
                  metrics=[masked_acc, masked_loss])
    
    vocab_size = 1.0 * target_text_processor.vocabulary_size()

    print('Expected_loss',  tf.math.log(vocab_size).numpy())
    print("Expected_acc", 1 / vocab_size)

    print(model.evaluate(val_ds, steps=20, return_dict=True))
    
    history = model.fit(
        train_ds.repeat(),
        epochs=100,
        steps_per_epoch=100,
        validation_data=val_ds,
        validation_steps=20,
        callbacks=[keras.callbacks.EarlyStopping(patience=10)])
    
    pyplot.plot(history.history['loss'], label='loss')
    pyplot.plot(history.history['val_loss'], label='val_loss')
    pyplot.ylim([0, max(pyplot.ylim())])
    pyplot.xlabel('Epoch #')
    pyplot.ylabel('CE/token')
    pyplot.legend()
    pyplot.show()
    
    pyplot.plot(history.history['masked_acc'], label='accuracy')
    pyplot.plot(history.history['val_masked_acc'], label='val_accuracy')
    pyplot.ylim([0, max(pyplot.ylim())])
    pyplot.xlabel('Epoch #')
    pyplot.ylabel('CE/token')
    pyplot.legend()
    pyplot.show()
    
    result = model.translate(['¿Todavía está en casa?']) # Are you still home
    print(result[0].numpy().decode())

    # Are you still home?
    model.plot_attention('¿Todavía está en casa?') 
    # This is my life.
    model.plot_attention('Esta es mi vida.')
    # Try to find out.
    model.plot_attention('Tratar de descubrir.')

    inputs = [
        'Hace mucho frio aqui.', # "It's really cold here."
        'Esta es mi vida.', # "This is my life."
        'Su cuarto es un desastre.' # "His room is a mess"
    ]
    result = model.translate(inputs)
    print(result[0].numpy().decode())
    print(result[1].numpy().decode())
    print(result[2].numpy().decode())

