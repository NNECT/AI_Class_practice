import re
import requests


def kor_to_alphabet_key(word):
    url = 'http://openhangul.com/nlp_ko2en?q={}'.format(word)
    return re.findall(r'<img src="images/cursor\.gif"><br>(.+)', requests.get(url).content.decode('utf-8'))[0].strip()


if __name__ == '__main__':
    word = '한글'
    print(kor_to_alphabet_key(word))
