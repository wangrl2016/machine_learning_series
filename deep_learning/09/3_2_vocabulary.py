import re

corpus_dataset = [
    'I drink and I know things.',
    'When you play the game of thrones, you win or you die.',
    'The true enemy won\'t wait out the storm, he brings the storm.'
]

if __name__ == '__main__':
    # 使用集合来去重
    unique_words = set()
    for sentence in corpus_dataset:
        sentence = re.sub(r'[.,]', '', sentence)
        words = sentence.split()
        unique_words.update(words)
    print(unique_words)
    print(len(unique_words))

