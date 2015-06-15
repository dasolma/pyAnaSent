__author__ = 'dasolma'
import pickle

class ModelAnalizer(object):

    def __init__(self, pipelines, parameters = None, n_jobs = 1):
        self.pipelines = pipelines
        self.classifiers = {}
        self.params = parameters
        self.n_jobs = n_jobs
        self.best_classifier = None
        self.best_name_classifier = None
        self.best_score = 0


    def fit(self, train_data, target_data):
        from sklearn.grid_search import GridSearchCV
        from sklearn.pipeline import Pipeline

        self.classifiers = {}
        if not self.params is None:
            print("Searching the best parameters set:")
        for name, pipeline in self.pipelines:
            if not self.params is None:
                params = self.get_params(pipeline)
                classifier = GridSearchCV(Pipeline(pipeline),  param_grid=params, n_jobs=self.n_jobs)
            else:
                classifier = Pipeline(pipeline)

            self.classifiers[name] = classifier.fit(train_data, target_data)

            if not self.params is None:
                print("\t" + name + ":" +  str(classifier.best_params_))

        print ""

    def get_params(self, pipeline):

        prefix = [name+"__" for name, proc in pipeline]
        params =  [(name, param) for name,param in self.params.iteritems() if name.startswith(tuple(prefix))]

        return dict(params)


    def predict(self, data):
        return self.best_classifier.predict(data)

    def score(self, test_data, test_target):
        from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score


        print "\nResuls:\n"
        row_format ="{:>20}" * 4
        row_format = "{:<45}" + row_format
        print row_format.format("", "f1", "recall", "precision", "accuracy")
        for name, classifier in self.classifiers.iteritems():

            pred = classifier.predict(test_data)
            f1 = f1_score(pred, test_target)
            recall = recall_score(pred, test_target)
            precision = precision_score(pred, test_target)
            accuracy = accuracy_score(pred, test_target)

            if accuracy > self.best_score:
                self.best_score, self.best_classifier, self.best_name_classifier = accuracy, classifier, name

            print row_format.format(name, f1, recall, precision, accuracy)


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



