import pickle

data = [
    (
        'address',
        {"ip": "8.8.8.8"}
    ),
    (
        'headers',
        {
            "Accept-Language": "en-US,en;q=0.8",
            "Host": "headers.jsontest.com",
            "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }
    ),
    (
        'datatime',
        {
           "time": "03:53:25 AM",
           "milliseconds_since_epoch": 1362196405309,
           "date": "03-02-2013"
        }
    )
]

filename = 'sample.pickle'
with open(filename, 'wb') as f:
    pickle.dump(data, f)

with open(filename, 'rb') as f:
    sample = pickle.load(f)

print(sample)
