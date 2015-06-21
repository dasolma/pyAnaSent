__author__ = 'dasolma'
import pickle
from config import *
from dataretriever import *
import matplotlib.pyplot as plt
import collections

class ModelAnalizer(object):

    def __init__(self, pipelines, parameters = None, n_jobs = 1):
        self.pipelines = pipelines
        self.classifiers = {}
        self.params = parameters
        self.n_jobs = n_jobs
        self.best_classifier = None
        self.best_name_classifier = None
        self.best_score = 0


    def fit(self, train_data, target_data, verbose=True):
        from sklearn.grid_search import GridSearchCV
        from sklearn.pipeline import Pipeline

        self.classifiers = {}
        if not self.params is None and verbose:
            print("Searching the best parameters set:")
        for name, pipeline in self.pipelines:
            if not self.params is None:
                params = self.get_params(pipeline)
                classifier = GridSearchCV(Pipeline(pipeline),  param_grid=params, n_jobs=self.n_jobs)
            else:
                classifier = Pipeline(pipeline)

            self.classifiers[name] = classifier.fit(train_data, target_data)

            if not self.params is None and verbose:
                print("\t" + name + ":" +  str(classifier.best_params_))

        if verbose: print ""

    def get_params(self, pipeline):

        prefix = [name+"__" for name, proc in pipeline]
        params =  [(name, param) for name,param in self.params.iteritems() if name.startswith(tuple(prefix))]

        return dict(params)


    def predict(self, data):
        return self.best_classifier.predict(data)

    def score(self, test_data, test_target, verbose = True):
        result = {}
        from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


        if verbose: print "\nResuls:\n"
        row_format ="{:>20}" * 4
        row_format = "{:<45}" + row_format
        if verbose: print row_format.format("", "f1", "recall", "precision", "accuracy")
        for name, classifier in self.classifiers.iteritems():

            pred = classifier.predict(test_data)
            f1 = f1_score(pred, test_target)
            recall = recall_score(pred, test_target)
            precision = precision_score(pred, test_target)
            accuracy = accuracy_score(pred, test_target)

            if accuracy > self.best_score:
                self.best_score, self.best_classifier, self.best_name_classifier = accuracy, classifier, name


            result[name] = (f1, recall, precision, accuracy)

            if verbose: print row_format.format(name, f1, recall, precision, accuracy)

        return result

    def cross_validation(self, data, target, cv=5):
        from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
        from sklearn import cross_validation
        from sklearn.pipeline import Pipeline
        from sklearn.grid_search import GridSearchCV
        from sklearn.cross_validation import ShuffleSplit
        import numpy as np

        print "\nResuls:\n"
        row_format ="{:>20}" * 4
        row_format = "{:<45}" + row_format
        print row_format.format("", "f1", "recall", "precision", "accuracy")

        for name, pipeline in self.pipelines:
            if not self.params is None:
                params = self.get_params(pipeline)
                classifier = GridSearchCV(Pipeline(pipeline),  param_grid=params,  n_jobs=self.n_jobs,
                                           cv=ShuffleSplit(n=len(target), train_size=int(len(target)*0.75),
                                            n_iter=3, random_state=1))

                classifier.fit(data, target)
                pred = classifier.predict(data)

            else:
                classifier = Pipeline(pipeline)
                pred = cross_validation.cross_val_predict(classifier, np.array(data), target, cv=cv)

            f1 = f1_score(pred, target)
            recall = recall_score(pred, target)
            precision = precision_score(pred, target)
            accuracy = accuracy_score(pred, target)


            if accuracy > self.best_score:
                self.best_score, self.best_classifier, self.best_name_classifier = accuracy, classifier, name

            print row_format.format(name, f1, recall, precision, accuracy)


    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    @staticmethod
    def load(path):
        return pickle.load(open(path, "rb"))



class StatisticsModelAnalizer():

    @staticmethod
    def create_graphs(tag_graph_indexes=(0,), iterations = 30, series_length = 10, title="" ):
        import config

        line_styles=["-", "--", ":", "-."]
        series = collections.defaultdict(lambda: [0] * series_length)
        x_axis = [i/float(series_length) for i in  range(1,series_length+1)]
        for j in range(iterations):
            for size in reversed(x_axis):

                data_train, data_test, target_train, target_test = get_data_ds2(size=size)
                ma = ModelAnalizer(config.pipelines, params)
                ma.fit(data_train, target_train, verbose=True)
                scores = ma.score(data_test, target_test, verbose=True)

                #accumulate scores
                for name,score in scores.iteritems():
                    names = name.split("->")
                    if len(tag_graph_indexes) == 1: name = names[tag_graph_indexes[0]]
                    else: name = ' -> '.join(names[tag_graph_indexes[0]:tag_graph_indexes[1]])

                    series[name][x_axis.index(size)] += score[3]


        plt.figure()
        min_y = 1.0
        for i,k in zip(range(len(series.keys())),series):
            ls = i / len(line_styles)
            series[k] = [v/iterations for v in series[k]]
            min_y = min(min_y, min(series[k]))
            plt.plot(x_axis, series[k], label=k, linestyle=line_styles[ls])

        plt.ylim(ymin=min_y)
        lgd = plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("%s (%d iterations)"%(title, iterations))
        graphname = ''.join(title.split())
        plt.savefig("graph%s.png"%graphname, bbox_extra_artists=(lgd,), bbox_inches='tight')


