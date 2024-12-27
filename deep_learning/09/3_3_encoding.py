
vocab_set = {
    'true', 'thrones', "won't", 'know', 'wait', 'I',
    'drink', 'brings', 'things', 'play', 'When', 'out',
    'storm', 'die', 'and', 'he', 'game', 'The',
    'the', 'win', 'or', 'enemy', 'you', 'of'
}

if __name__ == '__main__':
    vocab_dict = { word: index for index, word in enumerate(vocab_set, start=1)}
    print(vocab_dict)
