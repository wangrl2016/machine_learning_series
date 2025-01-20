import csv
import itertools
import nltk

vocabulary_size = 8000
unknown_token = 'UNKNOWN_TOKEN'
sentence_start_token = 'SENTENCE_START'
sentence_end_token = 'SENTENCE_END'

if __name__ == '__main__':
    nltk.download('punkt_tab')
    # Read the data and append SENTENCE_START and SENTENCE_END tokens.
    print('Reading CSV files...')
    with open('temp/reddit-comments-2015-08.csv', 'r') as f:
        reader = csv.reader(f, skipinitialspace=True)
        # Split full comments into sentences.
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in reader])
        # Append SENTENCE_START and SENTENCE_END.
        sentences = ['%s %s %s' % (sentence_start_token, x, sentence_end_token) for x in sentences]
    print('Parsed', len(sentences, 'sentences'))
