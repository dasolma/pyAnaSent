__author__ = 'dasolma'

from config import *
from modelanalizer import ModelAnalizer, StatisticsModelAnalizer
from dataretriever import *
from pipelines import compose_pipelines
import prepro
#get_data_ds1("data/SentimentClassification.txt")

StatisticsModelAnalizer.create_graphs((2,-1), title="Regression Logistic")

'''
data_train, data_test, target_train, target_test = get_data_ds2(size=0.9)
ma = ModelAnalizer(pipelines, params)
ma.fit(data_train, target_train)
ma.score(data_test, target_test)

data_train, data_test, target_train, target_test = get_data_ds2(size=0.9)
ma = ModelAnalizer(pipelines, params)
ma.fit(data_train, target_train)
ma.score(data_test, target_test)
'''