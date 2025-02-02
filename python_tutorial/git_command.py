import subprocess

if __name__ == '__main__':
    subprocess.run(['git', 'add', '.'])
    subprocess.run(['git', 'commit', '--amend', '--no-edit'])
    subprocess.run(['git', 'push', '-f'])
