__author__ = 'dasolma'

from config import *
from modelanalizer import ModelAnalizer
from dataretriever import *
from pipelines import compose_pipelines
#get_data_ds1("data/SentimentClassification.txt")
data_train, data_test, target_train, target_test = get_data_ds2()


ma = ModelAnalizer(pipelines, params)


ma.fit(data_train, target_train)
ma.score(data_test, target_test)