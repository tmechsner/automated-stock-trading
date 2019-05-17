import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from predictors.AbstractPredictor import AbstractPredictor


class KNNPredictor(AbstractPredictor):
    def __init__(self, k=5, weights='uniform'):
        super(KNNPredictor, self).__init__()
        self._k = k
        self._weights = weights
        self._classifier = KNeighborsClassifier(n_neighbors=self._k, weights=self._weights)
        self._trained_samples = 0
        self._predict_probas = []

    def predict(self, training_data, training_labels, todays_features, training_years, trading_days_per_year) -> int:
        new_samples_since_training = len(training_labels[self._trained_samples:])
        if new_samples_since_training >= training_years * trading_days_per_year:
            self._classifier.fit(training_data, training_labels)
            self._trained_samples = len(training_labels)

        self._predict_probas.append(self._classifier.predict_proba([todays_features])[0, 1])
        return self._classifier.predict([todays_features])[-1]

    def tune(self, stock_data, symbols, num_features, measure='f1', trading_frequency=10, training_years=3, trading_days_per_year=246):
        k_min = 5
        k_max = 21
        param_space = range(k_min, k_max + 1)[::2]
        accuracies = []
        f_scores = []
        params = []
        for weights in ['uniform', 'distance']:
            for k in param_space:
                accs = []
                fs = []
                for stock in symbols:
                    self._k = k
                    self._trained_samples = 0
                    self._classifier = KNeighborsClassifier(n_neighbors=self._k, weights=weights)

                    preds = []

                    X = stock_data[stock][['I{}'.format(x) for x in range(1, num_features + 1)]]
                    y = stock_data[stock]['target']
                    first_trading_day = training_years * trading_days_per_year + trading_frequency
                    last_trading_day = first_trading_day - trading_frequency
                    for trading_day in range(first_trading_day, len(y)):
                        preds.append(self.predict(X.iloc[:last_trading_day].values, y.iloc[:last_trading_day].values, X.iloc[trading_day], training_years, trading_days_per_year))
                        last_trading_day = trading_day
                    preds = pd.Series(preds, name='Predictions')

                    accuracy = metrics.accuracy_score(y.iloc[first_trading_day:], preds)
                    accs.append(accuracy)

                    f_score = metrics.f1_score(y.iloc[first_trading_day:], preds)
                    fs.append(f_score)

                mean_acc = np.array(accs).mean()
                accuracies.append(mean_acc)

                mean_f_score = np.array(fs).mean()
                f_scores.append(mean_f_score)

                params.append((k, weights))
                print('\nkNN  %s, k=%i' % (weights, k))
                print('kNN    Accuracy: %.3f' % mean_acc)
                print('kNN    F1 Score: %.3f' % mean_f_score)
        accuracies = np.array(accuracies)
        f_scores = np.array(f_scores)
        print('\nkNN FINAL RESULTS')
        if measure == 'accuracy':
            argmax = accuracies.argsort()[-3:]
            for i in argmax:
                (k, weights) = params[i]
                print('kNN  Best result: accuracy %.3f (%s, k=%i)' % (accuracies.max(), weights, k))
        elif measure == 'f1':
            argmax = f_scores.argsort()[-3:]
            for i in argmax:
                (k, weights) = params[i]
                print('kNN  Best result: f score %.3f (%s, k=%i)' % (f_scores.max(), weights, k))
        else:
            raise NotImplementedError
