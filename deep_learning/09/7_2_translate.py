import os
import keras
import re
import pathlib
import string
import random
import tensorflow

os.environ['KERAS_BACKEND'] = 'tensorflow'

# English-to-Spanish translation with a sequence-to-sequence Transformer.

def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return (
        {
            "encoder_inputs": eng,
            "decoder_inputs": spa[:, :-1],
        },
        spa[:, 1:],
    )

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tensorflow.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.cache().shuffle(2048).prefetch(16)

class TransformerEncoder(keras.layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                keras.layers.Dense(dense_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        if mask is not None:
            padding_mask = keras.ops.cast(mask[:, None, :], dtype="int32")
        else:
            padding_mask = None

        attention_output = self.attention(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = keras.layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = keras.ops.shape(inputs)[-1]
        positions = keras.ops.arange(0, length, 1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return keras.ops.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config


class TransformerDecoder(keras.layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                keras.layers.Dense(latent_dim, activation="relu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = keras.layers.LayerNormalization()
        self.layernorm_2 = keras.layers.LayerNormalization()
        self.layernorm_3 = keras.layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        inputs, encoder_outputs = inputs
        causal_mask = self.get_causal_attention_mask(inputs)

        if mask is None:
            inputs_padding_mask, encoder_outputs_padding_mask = None, None
        else:
            inputs_padding_mask, encoder_outputs_padding_mask = mask

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask,
            query_mask=inputs_padding_mask,
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            query_mask=inputs_padding_mask,
            key_mask=encoder_outputs_padding_mask,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = keras.ops.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = keras.ops.arange(sequence_length)[:, None]
        j = keras.ops.arange(sequence_length)
        mask = keras.ops.cast(i >= j, dtype="int32")
        mask = keras.ops.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = keras.ops.concatenate(
            [keras.ops.expand_dims(batch_size, -1), keras.ops.convert_to_tensor([1, 1])],
            axis=0,
        )
        return keras.ops.tile(mask, mult)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

if __name__ == '__main__':
    text_file = keras.utils.get_file(
        fname='spa-end.zip',
        origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip',
        extract=True)
    text_file = pathlib.Path(text_file).parent / 'spa-end_extracted' / 'spa-eng' / 'spa.txt'

    with open(text_file) as f:
        lines = f.read().split('\n')[:-1]
    text_pairs = []
    for line in lines:
        eng, spa = line.split('\t')
        spa = '[start] ' + spa + ' [end]'
        text_pairs.append((eng, spa))
    
    # Here's what our sentence pairs look like.
    # ('Give it to him.', '[start] Dáselo a él. [end]')
    # ('He lost face.', '[start] Él perdió prestigio. [end]')
    # ('Tom decided to protest.', '[start] Tom decidió protestar. [end]')
    for _ in range(5):
        print(random.choice(text_pairs))

    # Split the sentence pairs into a trianing set, a validation set, and a test set.
    random.shuffle(text_pairs)
    num_validate_samples = int(0.15 * len(text_pairs))
    num_train_samples = len(text_pairs) - 2 * num_validate_samples
    train_pairs = text_pairs[0:num_train_samples]
    val_pairs = text_pairs[num_train_samples:num_train_samples+num_validate_samples]
    test_pairs = text_pairs[num_train_samples + num_validate_samples:]

    print(str(len(text_pairs)), 'total pairs')
    print(str(len(train_pairs)), 'training pairs')
    print(str(len(val_pairs)), 'valication pairs')
    print(str(len(test_pairs)), 'test pairs')

    vocab_size = 15000
    sequence_length = 20
    batch_size = 64

    eng_vectorization = keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length
    )

    strip_chars = string.punctuation + "¿"
    strip_chars = strip_chars.replace("[", "")
    strip_chars = strip_chars.replace("]", "")
    def custom_standardization(input_string):
        lowercase = tensorflow.strings.lower(input_string)
        return tensorflow.strings.regex_replace(lowercase, '%s' % re.escape(strip_chars), '')
    spa_vectorization = keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization,)
    
    train_eng_texts = [pair[0] for pair in train_pairs]
    train_spa_texts = [pair[1] for pair in train_pairs]
    eng_vectorization.adapt(train_eng_texts)
    spa_vectorization.adapt(train_spa_texts)

    train_ds = make_dataset(train_pairs)
    val_ds = make_dataset(val_pairs)

    for inputs, targets in train_ds.take(1):
        print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
        print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
        print(f"targets.shape: {targets.shape}")

    embed_dim = 256
    latent_dim = 2048
    num_heads = 8

    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads)([x, encoder_outputs])
    x = keras.layers.Dropout(0.5)(x)
    decoder_outputs = keras.layers.Dense(vocab_size, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

    transformer = keras.Model(
        {"encoder_inputs": encoder_inputs, "decoder_inputs": decoder_inputs},
        decoder_outputs,
        name="transformer",
    )

    epochs = 1  # This should be at least 30 for convergence
    transformer.summary()

    transformer.compile(
        "rmsprop",
        loss=keras.losses.SparseCategoricalCrossentropy(ignore_class=0),
        metrics=["accuracy"],
    )
    transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)

    # prediction
    # We simply feed into the model the vectorized English sentence as well as the
    # target token '[start]', then we repeatedly generated the next token, until we hit
    # the token '[end]'.
    spa_vocab = spa_vectorization.get_vocabulary()
    spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
    max_decoded_sentence_length = 20

    def decode_sequence(input_sentence):
        tokenized_input_sentence = eng_vectorization([input_sentence])
        decoded_sentence = '[start]'
        for i in range(max_decoded_sentence_length):
            tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
            predictions = transformer(
                {
                    "encoder_inputs": tokenized_input_sentence,
                    "decoder_inputs": tokenized_target_sentence,
                }
            )

            # ops.argmax(predictions[0, i, :]) is not a concrete value for jax here
            sampled_token_index = keras.ops.convert_to_numpy(
                keras.ops.argmax(predictions[0, i, :])
            ).item(0)
            sampled_token = spa_index_lookup[sampled_token_index]
            decoded_sentence += " " + sampled_token

            if sampled_token == "[end]":
                break
        return decoded_sentence
    
    for _ in range(30):
        text_pair = random.choice(text_pairs)
        print('Src', text_pair)
        translated = decode_sequence(text_pair[0])
        print('Pred', translated)
