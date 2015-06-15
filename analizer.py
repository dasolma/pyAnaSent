__author__ = 'dasolma'

from config import *
from modelanalizer import ModelAnalizer
from dataretriever import get_data
from pipelines import compose_pipelines
data_train, data_test, target_train, target_test = get_data("data/SentimentClassification.txt")


ma = ModelAnalizer(pipelines, params)


ma.fit(data_train, target_train)
ma.score(data_test, target_test)