__author__ = 'dasolma'
from sklearn.cross_validation import train_test_split


class TweetDataSetReader():

    def __init__(self, test_size=0.75):
        self.base_data_dir = 'data/tweets/'
        self.classes = []
        self.tweets = []
        self.test_size = test_size

        self.read_corpus()
        self.tweets_train, self.tweets_test, self.classes_train, self.classes_test = \
            train_test_split(self.tweets, self.classes, test_size=self.test_size)


    def read_corpus(self):
        import json
        import os.path
        self.classes = []
        count = 0
        for l in open(self.base_data_dir+"corpus.csv").readlines():
            l = l.replace("\"", "").split(",")
            tweet_file = self.base_data_dir+"rawdata/"+l[2].strip()+".json"

            if os.path.isfile(tweet_file):
                self.classes.append(l)

                self.tweets.append(
                    json.load(
                        open(tweet_file)
                    )
                )

            else:
                count += 1
                #print "tweet: %s , %s not found"%(l[1], tweet_file)
                pass

        print "Found: %d tweets"%len(self.tweets)
        print "Not found: %d but references in corpus"%count

    def get_all(self):

        classes_train = []
        for clas, tweet in zip(self.classes_train, self.tweets_train):
                classes_train.append(
                    2.0 if clas[1] == 'irrelevant' else \
                    0.0 if clas[1] == 'neutral' else \
                    -1.0 if clas[1] == 'negative' else \
                    1.0)

        classes_test = []
        for clas, tweet in zip(self.classes_test, self.tweets_test):
                classes_test.append(
                    2.0 if clas[1] == 'irrelevant' else \
                    0.0 if clas[1] == 'neutral' else \
                    -1.0 if clas[1] == 'negative' else \
                    1.0)

        return self.tweets_train, self.tweets_test, classes_train, classes_test

    def get_dataset_irrelevantvssentiment(self):

        classes_train = []
        for clas, tweet in zip(self.classes_train, self.tweets_train):
            classes_train.append( 0.0 if clas[1] == 'irrelevant'  else 1.0 )

        classes_test = []
        for clas, tweet in zip(self.classes_test, self.tweets_test):
            classes_test.append( 0.0 if clas[1] == 'irrelevant'  else 1.0 )

        return self.tweets_train, self.tweets_test, classes_train, classes_test

    def get_dataset_posneg(self):

        clas_train = []
        tweets_train = []
        for clas, tweet in zip(self.classes_train, self.tweets_train):
            if  not clas[1] in ('irrelevant', 'neutral'):
                clas_train.append(
                    1 if clas[1] == 'positive' else \
                    0 if clas[1] == 'negative' else \
                    -1.0)
                tweets_train.append(tweet['text'])

        clas_test = []
        tweets_test = []
        for clas, tweet in zip(self.classes_test, self.tweets_test):
            if not clas[1] in ('irrelevant', 'neutral') :
                clas_test.append(
                    1 if clas[1] == 'positive' else \
                    0 if clas[1] == 'negative' else \
                    -1.0)
                tweets_test.append(tweet['text'])

        return tweets_train, tweets_test, clas_train, clas_test

    def get_by_type(self, type):

        tweets = []
        for clas, tweet in zip(self.classes, self.tweets):
            if clas[1] == type: tweets.append(tweet)

        return tweets




