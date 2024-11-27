import requests

time_machine_text_url = 'https://www.gutenberg.org/cache/epub/35/pg35.txt'

if __name__ == '__main__':
    response = requests.get(time_machine_text_url)
    if response.status_code == 200:
        lines = response.text.splitlines()
        print('Total line:', len(lines))
        print('Line 1:', lines[0])
        print('Line 10:',lines[10])
    else:
        print('Download error, status code:', response.status_code)  
