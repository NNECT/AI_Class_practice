import random
import re
import requests
from download import download
import csv

url = 'http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stdId=108'
filename = 'weather.xml'
download(url, filename, 'utf-8')
with open(filename, encoding='utf-8') as f:
    text = f.read()

# url = 'http://www.kma.go.kr/weather/forecast/mid-term-rss3.jsp?stdId=108'
# res = requests.get(url)
# res.encoding = 'utf-8'
# text = res.text

# DOTALL 여러 줄에 걸쳐있을 때!
body = re.findall(r'<body>(.+?)</body>', text, re.DOTALL)
location = []
for b in body:
    location += re.findall(r'<location.*?>(.+?)</location>', b, re.DOTALL)


# city = []
# for loc in location:
#     city += re.findall(r'<city>(.+?)</city>', loc)
#

f = open('WeatherData.csv', 'w', encoding='utf-8')

for loc in location:
    pc = re.findall(r'<province>(.+?)</province>.+?<city>(.+?)</city>', loc, re.DOTALL)
    data = re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)
    for d in data:
        items = re.findall(r'>(.+?)</', d)
        f.write(','.join(list(pc[0]) + items))
        f.write('\n')

f.close()

# words = ['yellow', 'hi']
# length = sum([len(w) for w in words])
# a = [random.randrange(100) for _ in range(10)]
# num = max([n for n in a if n % 2])
# print(num)
#
# b1 = [random.randrange(100) for _ in range(10)]
# b2 = [random.randrange(100) for _ in range(10)]
# b3 = [random.randrange(100) for _ in range(10)]
# b = [b1, b2, b3]
# # result = sum([j for i in b for j in i if j % 2])
# result = [[j for j in i if j % 2] for i in b]
# print(result)

#
# result = []
# for loc in location:
#     province = re.findall(r'<province>(.+?)</province>', loc)[0]
#     city = re.findall(r'<city>(.+?)</city>', loc)[0]
#     data = [dict(re.findall(r'<(.+?)>(.+?)<', d)) for d in re.findall(r'<data>(.+?)</data>', loc, re.DOTALL)]
#     result.append([province, city, data])
#