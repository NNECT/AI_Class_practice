import os
import requests


def download(url, filename, encoding='utf-8'):
    if not os.path.exists(filename):
        with open(filename, "w", encoding=encoding) as f:
            res = requests.get(url)
            res.encoding = encoding
            f.write(res.text)
    else:
        print(f'{filename}: already downloaded')