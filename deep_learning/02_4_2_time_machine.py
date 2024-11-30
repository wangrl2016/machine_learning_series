import collections
import requests

time_machine_text_url = 'https://www.gutenberg.org/cache/epub/35/pg35.txt'

def tokenize(lines, token='word'):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('Tokenize error')

def count_corpus(tokens):
    if len(tokens) == 0 or isinstance(tokens[0], list):
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    def __init__(self, tokens=None, min_freq = 1):
        if tokens is None:
            tokens = []
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        # 未知词元的索引为 0
        self.idx_to_token = ['<unknown>']
        # { '<unknown>': 0, }
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unknown)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unknown(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs

if __name__ == '__main__':
    response = requests.get(time_machine_text_url)
    response.encoding = 'utf-8'
    if response.status_code == 200:
        lines = response.text.splitlines()
        print('Total line:', len(lines))
        print('Line 1:', lines[0])
        print('Line 10:',lines[10])

        tokens = tokenize(lines)
        for i in range(10):
            print(tokens[i])

        vocab = Vocab(tokens)
        print(list(vocab.token_to_idx.items())[0:10])

        for i in [0, 10]:
            print('文本:', tokens[i])
            print('索引:', vocab[tokens[i]])
    else:
        print('Download error, status code:', response.status_code)  
