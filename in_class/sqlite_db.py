import sqlite3


def create_db():
    conn = sqlite3.connect('WeatherData.sqlite')
    cur = conn.cursor()

    query = 'CREATE TABLE WeatherData (prov TEXT, city TEXT, mode TEXT, tmEf TEXT, wf TEXT, tmn TEXT, tmx TEXT, rnSt TEXT);'
    cur.execute(query)

    conn.commit()
    conn.close()


def insert_row(row):
    conn = sqlite3.connect('WeatherData.sqlite')
    cur = conn.cursor()

    query = 'INSERT INTO WeatherData VALUES({});'.format(', '.join([f'"{i}"' for i in row]))
    cur.execute(query)

    conn.commit()
    conn.close()


def insert_all(rows):
    conn = sqlite3.connect('WeatherData.sqlite')
    cur = conn.cursor()

    for row in rows:
        query = 'INSERT INTO WeatherData VALUES({});'.format(', '.join([f'"{i}"' for i in row]))
        cur.execute(query)

    conn.commit()
    conn.close()


def fetch_all():
    conn = sqlite3.connect('WeatherData.sqlite')
    cur = conn.cursor()

    query = 'SELECT * FROM WeatherData'
    records = [record for record in cur.execute(query)]

    conn.commit()
    conn.close()

    return records


def search_city(city):
    conn = sqlite3.connect('WeatherData.sqlite')
    cur = conn.cursor()

    query = 'SELECT * FROM WeatherData WHERE city="{}"'.format(city)
    records = [record for record in cur.execute(query)]

    conn.commit()
    conn.close()

    return records


create_db()
with open('WeatherData.csv', encoding='utf-8') as f:
    rows = [line.strip().split(',') for line in f.readlines()]
insert_all(rows)
rows = search_city("전주")
print(*rows, sep='\n')
