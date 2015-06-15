__author__ = 'dasolma'
from sklearn.cross_validation import train_test_split

def get_data(path):

    data = []
    target = []
    for line in open(path).readlines():
        t, d = line.split("\t")
        data.append(d.strip().decode('utf-8'))
        target.append(int(t))

    return train_test_split(data, target, test_size=0.66, random_state=42)

get_data("data/SentimentClassification.txt")