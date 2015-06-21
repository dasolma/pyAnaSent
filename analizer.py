__author__ = 'dasolma'

from config import *
import config
from modelanalizer import ModelAnalizer, StatisticsModelAnalizer
from dataretriever import *
from pipelines import compose_pipelines
import prepro
#get_data_ds1("data/SentimentClassification.txt")

for classifier in classifiers:
    print  class_names[classifier[0]]
    config.pipelines =  name_pipelines(compose_pipelines(transformers, [classifier]))
    StatisticsModelAnalizer.create_graphs((3,-1), title="Prepros with " + class_names[classifier[0]])
