__author__ = 'dasolma'
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from prepro import Lematization, POS_tag, Steammer,EmoticonsReplacer, LinkRemover
from pipelines import *

transformers = [
      [ ("emo", EmoticonsReplacer())],
      [ None, ("link", LinkRemover())],
      [ ('postag', POS_tag() ) ],
      #[ None,  ('lema', Lematization() ), ('stem', Steammer()) ],
      [ ('vect_ngram1', CountVectorizer(ngram_range=(1,1))),
         ('vect_ngram2', CountVectorizer(ngram_range=(1,2))),
         ('vect_ngram3', CountVectorizer(ngram_range=(1,3))),
      ],
      #[('tfidf', TfidfTransformer())]

     ]

classifiers = [('mNB', MultinomialNB()),
               #('svm', LinearSVC()),
               #('LR', LogisticRegression()),
               #('SGD', SGDClassifier('perceptron')),
               #('Random forest', RandomForestClassifier())
              ]

params = {#'vect__ngram_range':[(1,1),(1,2),(1,3)],
          'tfidf__smooth_idf':[True, False],
          'tfidf__use_idf': [True, False],
          'tfidf__sublinear_tf': [True, False],
          'mNB__alpha':[0, 0.1, 0.5, 1.0],
          'svm__C':[0.1, 10, 100],
          'tfidf+ling__sublinear_tf': [True, False],
          'tfidf+ling__ngram_range':[(1,1),(1,2),(1,3)],
          'tfidf+ling__smooth_idf':[True, False],
          'tfidf+ling__use_idf': [True, False],
          }



pipelines =  name_pipelines(compose_pipelines(transformers, classifiers))
