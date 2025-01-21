import csv
import itertools
import nltk
import numpy

vocabulary_size = 8000
unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'

if __name__ == '__main__':
    nltk.download('punkt_tab')
    # Read the data and append SENTENCE_START and SENTENCE_END tokens.
    print('Reading CSV files...')
    with open('temp/reddit-comments-2015-08.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences.
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END.
        sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print('Parsed', len(sentences), 'sentences')
    
    # Tokenize the sentences into words.
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # Count the word frequencies.
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    print('Found', len(word_freq.items()), 'unique words tokens')

    # Get the most common words and build index_to_word and word_to_index vectors.
    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    print('Using vocabulary size', vocabulary_size)
    print('The least frequent word in our vocabulary is', vocab[-1][0], 'and appeared', vocab[-1][1], 'times')

    # Replace all words in our vocabulary with the unkown token.
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]
    
    print('Example sentence 10:', sentences[10])
    print('Example sentence 10 after pre-processing:', tokenized_sentences[10])
    
    # Create the training data.
    x_train = numpy.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences], dtype=object)
    y_train = numpy.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences], dtype=object)
    print('Sentence 10 source:', x_train[10])
    print('Sentence 10 target:', y_train[10])
