import sys

from math import sqrt

import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from predictors.AbstractPredictor import AbstractPredictor


class RFPredictor(AbstractPredictor):
    def __init__(self, n_estimators=10, max_features='auto', max_depth=None):
        super(RFPredictor, self).__init__()
        self._importances = np.zeros(22)
        self._n_estimators = n_estimators
        self._max_features = max_features
        self._max_depth = max_depth

        self._classifier = RandomForestClassifier(n_estimators=self._n_estimators,
                                                  max_features=self._max_features,
                                                  max_depth=self._max_depth,
                                                  random_state=42)
        self._trained_samples = 0
        self._predict_probas = []

    def feature_relevance(self):
        indices = np.argsort(self._importances)[::-1]

        # Print the feature ranking
        print("RF Feature ranking:")

        for f in range(len(indices)):
            print("%d. feature %d (%f)" % (f + 1, indices[f], self._importances[indices[f]]))

    def predict(self, training_data, training_labels, todays_features, training_years, trading_days_per_year) -> int:
        new_samples_since_training = len(training_labels[self._trained_samples:])
        if new_samples_since_training >= training_years * trading_days_per_year:
            self._classifier.fit(training_data, training_labels)
            self._trained_samples = len(training_labels)
            importances = self._classifier.feature_importances_
            for f in range(len(importances)):
                self._importances[f] += importances[f]

        self._predict_probas.append(self._classifier.predict_proba([todays_features])[0, 1])
        return self._classifier.predict([todays_features])[-1]

    def tune(self, stock_data, symbols, num_features, measure='f1', trading_frequency=10, training_years=3, trading_days_per_year=246):
        n_est_space = [10, 25, 50, 100]
        sqrt_feat = sqrt(num_features)
        max_f_space = [round(sqrt_feat) - 1, round(sqrt_feat), round(sqrt_feat) + 1]
        # max_d_space = [3, 5, 9, None]
        max_d_space = [None]
        accuracies = []
        f_scores = []
        params = []
        for n_estimators in n_est_space:
            for max_features in max_f_space:
                for max_depth in max_d_space:
                    accs = []
                    fs = []
                    for stock in symbols:
                        print(stock)
                        self._n_estimators = n_estimators
                        self._max_features = max_features
                        self._max_depth = max_depth
                        self._trained_samples = 0
                        self._classifier = RandomForestClassifier(n_estimators=self._n_estimators,
                                                                  max_features=self._max_features,
                                                                  max_depth=self._max_depth,
                                                                  random_state=42)

                        preds = []

                        X = stock_data[stock][['I{}'.format(x) for x in range(1, num_features + 1)]]
                        y = stock_data[stock]['target']
                        first_trading_day = training_years * trading_days_per_year + trading_frequency
                        last_trading_day = first_trading_day - trading_frequency
                        for trading_day in range(first_trading_day, len(y)):
                            sys.stdout.write('\r%i/%i' % (trading_day, len(y)))
                            sys.stdout.flush()
                            preds.append(self.predict(X.iloc[:last_trading_day].values, y.iloc[:last_trading_day].values, X.iloc[trading_day],
                                                      training_years, trading_days_per_year))
                            last_trading_day = trading_day
                        print('')
                        preds = pd.Series(preds, name='Predictions')

                        accuracy = metrics.accuracy_score(y.iloc[first_trading_day:], preds)
                        accs.append(accuracy)

                        f_score = metrics.f1_score(y.iloc[first_trading_day:], preds)
                        fs.append(f_score)

                    mean_acc = np.array(accs).mean()
                    accuracies.append(mean_acc)

                    mean_f_score = np.array(fs).mean()
                    f_scores.append(mean_f_score)

                    params.append((n_estimators, max_features, max_depth))
                    print('\nRF  n_estimators=%i max_features=%i max_depth=%s' % (n_estimators, max_features, max_depth))
                    print('RF    Accuracy: %.3f' % mean_acc)
                    print('RF    F1 Score: %.3f' % mean_f_score)
        accuracies = np.array(accuracies)
        f_scores = np.array(f_scores)
        print('\nRF FINAL RESULTS')
        if measure == 'accuracy':
            argmax = accuracies.argsort()[-3:]
            for i in argmax:
                (n_estimators, max_features, max_depth) = params[i]
                print('RF  Best result: accuracy %.3f (n_estimators=%i max_features=%i max_depth=%s)' % (accuracies[i], n_estimators, max_features, max_depth))
        elif measure == 'f1':
            argmax = f_scores.argsort()[-3:]
            for i in argmax:
                (n_estimators, max_features, max_depth) = params[i]
                print('RF Best result: f score %.3f (n_estimators=%i max_features=%i max_depth=%s)' % (f_scores.max[i], n_estimators, max_features, max_depth))
        else:
            raise NotImplementedError
