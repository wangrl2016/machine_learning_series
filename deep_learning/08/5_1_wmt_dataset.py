import tarfile
import re
import string
import unicodedata

def extract_tgz(file_path, output_dir):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(path=output_dir)

# Load doc into memory.
def load_doc(filename):
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close
    return text

# Split a loaded document into sentences.
def to_sentences(doc):
    return doc.strip().split('\n')

# Shortest and longest sentence lengths.
def sentence_lengths(sentences):
    lengths = [len(s.split()) for s in sentences]
    return min(lengths), max(lengths)

def clean_lines(lines):
    cleaned = list()
    # prepare regex for char filtering.
    re_print = re.compile('[^%s]' % re.escape(string.printable))
    # prepare translation table for removing punctutation
    # 删除所有标点符号
    table = str.maketrans('', '', string.punctuation)
    for line in lines:
        # Café déjà vu résumé naïve
        # Cafe deja vu resume naive
        line = unicodedata.normalize('NFD', line).encode('ascii', 'ignore')
        line = line.decode('utf-8')
        # tokenize on white space
        line = line.split()
        # convert to lower case
        line = [word.lower() for word in line]
        # remove punctutation from each token
        line = [word.translate(table) for word in line]
        # remove non-printable chars form each token
        line = [re_print.sub('', w) for w in line]
        # remove tokens with numbers in them
        line = [word for word in line if word.isalpha()]
        # store as string
        cleaned.append(' '.join(line))
    return cleaned

if __name__ == '__main__':
    extract_tgz('temp/fr-en.tgz', 'temp/fr-en')
    
    # load English data
    filename = 'temp/fr-en/europarl-v7.fr-en.en'
    doc = load_doc(filename)
    sentences = to_sentences(doc)
    minlen, maxlen = sentence_lengths(sentences)
    print('English data: sentences = %d, min = %d, max = %d' % (len(sentences), minlen, maxlen))

    # load France data
    # filename = 'temp/fr-en/europarl-v7.fr-en.fr'
    # doc = load_doc(filename)
    # sentences = to_sentences(doc)
    # minlen, maxlen = sentence_lengths(sentences)
    # print('France data: sentences = %d, min = %d, max = %d' % (len(sentences), minlen, maxlen))


