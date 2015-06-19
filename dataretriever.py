__author__ = 'dasolma'
from sklearn.cross_validation import train_test_split
from datatweets import TweetDataSetReader

def get_data_ds1(path):

    data = []
    target = []
    for line in open(path).readlines():
        t, d = line.split("\t")
        data.append(d.strip().decode('utf-8').lower())
        target.append(int(t))

    return train_test_split(data, target, test_size=0.66, random_state=42)

def get_data_ds2():

    dt = TweetDataSetReader()
    dt.read_corpus()
    return dt.get_dataset_posneg()