# coding: utf-8

__author__ = 'dasolma'
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer

cache = {}
lemmatizer = WordNetLemmatizer()

class LinkRemover(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):
            post = ' '.join(re.sub("(\w+:\/\/\S+)","",post).split())
            result[i] = post

        return result




class Lematization(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):
            key = "lm_" + str(id(post))
            if not key in cache: cache[key] = ' '.join([lemmatizer.lemmatize(w) for w in post.split()])
            post = cache[key]
            result[i] = post

        return result


class POS_tag(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        from nltk.tokenize import word_tokenize
        from nltk.tag import pos_tag

        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):
            key = "pt_" + str(id(post))
            if not key in cache: cache[key] = ' '.join(["%s %s"%(t,w)  for w,t in pos_tag(post.split())])
            #word_tokenize(post.decode('utf-8'))
            post = cache[key]
            result[i] = post

        return result


class Steammer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        steammer = SnowballStemmer("english")
        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):

            result[i] = " ".join([SnowballStemmer("english").stem(word) for word in post.split()])

        return result
