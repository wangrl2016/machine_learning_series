import os

def no_space(char, prev_char):
    return char in set(',.!?') and prev_char != ' '

if __name__ == '__main__':
    text = []
    local_file_path = './temp/fra.txt'
    with open(local_file_path, 'r', encoding='utf-8') as fp:
        raw_lines = fp.readlines()
        for index, line in enumerate(raw_lines):
            line = line.replace('\xa0', ' ').lower()
            parts = line.split('\t')
            if len(parts) == 3:
                print(parts[0], parts[1])
            else:
                print('Error line', line)
            if index > 10:
                break
