__author__ = 'dasolma'
from sklearn.cross_validation import train_test_split
from random import shuffle

class TweetDataSetReader():

    def __init__(self, test_size=0.25):
        self.base_data_dir = 'data/tweets/'
        self.test_size = test_size

        if not hasattr(TweetDataSetReader, "tweets"):
            print "Reading corpus"
            self.read_corpus()

            TweetDataSetReader.tweets_train, TweetDataSetReader.tweets_test, \
            TweetDataSetReader.classes_train, TweetDataSetReader.classes_test = \
                train_test_split(TweetDataSetReader.tweets, TweetDataSetReader.classes, test_size=self.test_size)


    def read_corpus(self):
        import json
        import os.path
        TweetDataSetReader.tweets = []
        TweetDataSetReader.classes = []
        count = 0
        for l in open(self.base_data_dir+"corpus.csv").readlines():
            l = l.replace("\"", "").split(",")
            tweet_file = self.base_data_dir+"rawdata/"+l[2].strip()+".json"

            if os.path.isfile(tweet_file):
                TweetDataSetReader.classes.append(l)

                TweetDataSetReader.tweets.append(
                    json.load(
                        open(tweet_file)
                    )
                )

            else:
                count += 1
                #print "tweet: %s , %s not found"%(l[1], tweet_file)
                pass

        print "Found: %d tweets"%len(TweetDataSetReader.tweets)
        print "Not found: %d but references in corpus"%count

    def get_all(self):

        classes_train = []
        for clas, tweet in zip(TweetDataSetReader.classes_train, TweetDataSetReader.tweets_train):
                classes_train.append(
                    2.0 if clas[1] == 'irrelevant' else \
                    0.0 if clas[1] == 'neutral' else \
                    -1.0 if clas[1] == 'negative' else \
                    1.0)

        classes_test = []
        for clas, tweet in zip(TweetDataSetReader.classes_test, TweetDataSetReader.tweets_test):
                classes_test.append(
                    2.0 if clas[1] == 'irrelevant' else \
                    0.0 if clas[1] == 'neutral' else \
                    -1.0 if clas[1] == 'negative' else \
                    1.0)

        return TweetDataSetReader.tweets_train, TweetDataSetReader.tweets_test, classes_train, classes_test

    def get_dataset_irrelevantvssentiment(self):

        classes_train = []
        for clas, tweet in zip(TweetDataSetReader.classes_train, TweetDataSetReader.tweets_train):
            classes_train.append( 0.0 if clas[1] == 'irrelevant'  else 1.0 )

        classes_test = []
        for clas, tweet in zip(TweetDataSetReader.classes_test, TweetDataSetReader.tweets_test):
            classes_test.append( 0.0 if clas[1] == 'irrelevant'  else 1.0 )

        return TweetDataSetReader.tweets_train, TweetDataSetReader.tweets_test, classes_train, classes_test

    def get_dataset_posneg(self, size = 1):

        clas_train = []
        tweets_train = []
        for clas, tweet in zip(TweetDataSetReader.classes_train, TweetDataSetReader.tweets_train):
            if  not clas[1] in ('irrelevant', 'neutral'):
                clas_train.append(
                    1 if clas[1] == 'positive' else \
                    0 if clas[1] == 'negative' else \
                    -1.0)
                tweets_train.append(tweet['text'])

        clas_test = []
        tweets_test = []
        for clas, tweet in zip(TweetDataSetReader.classes_test, TweetDataSetReader.tweets_test):
            if not clas[1] in ('irrelevant', 'neutral') :
                clas_test.append(
                    1 if clas[1] == 'positive' else \
                    0 if clas[1] == 'negative' else \
                    -1.0)
                tweets_test.append(tweet['text'])

        if size < 1:
            l = int(len(tweets_train) * size)
            index = range(len(tweets_train))
            shuffle(index)
            index = index[:l]
            tweets_train = [tweets_train[i] for i in index]
            clas_train = [clas_train[i] for i in index]
            #l = int(len(self.tweets_test) * size)
            #tweets_test = tweets_test[:l]
            #clas_test = clas_test[:l]


        return tweets_train, tweets_test, clas_train, clas_test

    def get_by_type(self, type):

        tweets = []
        for clas, tweet in zip(TweetDataSetReader.classes, TweetDataSetReader.tweets):
            if clas[1] == type: tweets.append(tweet)

        return tweets




