# coding: utf-8

__author__ = 'dasolma'
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem.snowball import SnowballStemmer
import numpy as np
import re
from nltk.stem.wordnet import WordNetLemmatizer

cache = {}
lemmatizer = WordNetLemmatizer()

emoticons = {"happy": ":-) :) :D :o) :] :3 :c) :> =] 8) =) :} :^) :っ) :-))",
             "laughing" : ":-D 8-D 8D x-D xD X-D XD =-D =D =-3 =3 B^D",
             "sad" : ">:[ :-( :(  :-c :c :-<  :っC :< :-[ :[ :{",
             "crying" : ":'-( :'( :,(",
             "surprise" : ">:O :-O :O :-o :o 8-0 O_O o-o O_o o_O o_o O-O",
             "kiss": ":* :^* ( '}{' )",
             "tongue" : ">:P :-P :P X-P x-p xp XP :-p :p =p :-Þ :Þ :þ :-þ :-b :b d:"}

abbreviations = ["RT"]

class EmoticonsReplacer(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):
            key = "emo_" + str(id(post))
            if not key in cache: cache[key] = self.replace_emoticons(post)
            post = cache[key]
            result[i] = post

        return result


    def replace_emoticons(self, text):


        for senti, emos in emoticons.iteritems():
            for emo in emos.split():
                text = text.replace(emo.decode('utf-8'), " "+senti+" ").replace("  ", " ")

        return text

class LinkRemover(BaseEstimator, TransformerMixin):

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):
            key = "link_" + str(id(post))
            if not key in cache: cache[key] = ' '.join(re.sub("(\w+:\/\/\S+)","",post).split())
            post = cache[key]
            result[i] = post


        return result

class AbbreviationRemover(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):
            key = "ar_" + str(id(post))

            if not key in cache:
                tokens =[token for token in post.split() if not token in abbreviations]
                cache[key] = ' '.join(tokens)
            post = cache[key]
            result[i] = post


        return result

class TagRemover(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        result = np.empty(shape=(len(posts)), dtype=object)

        for i, post in enumerate(posts):
            key = "tr_" + str(id(post))
            if not key in cache: cache[key] = ' '.join(re.sub("[\@\#]+","",post).split())
            post = cache[key]
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
            if not key in cache: cache[key] = ' '.join(["%s/%s"%(w,t)  for w,t in pos_tag(post.split())])
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
