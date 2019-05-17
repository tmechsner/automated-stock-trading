from math import sqrt

import numpy as np
import pandas as pd
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from predictors.AbstractPredictor import AbstractPredictor


class RFConfidencePredictor(AbstractPredictor):
    def __init__(self, buy_signal, sell_signal, keep_signal, n_estimators=10, max_features='auto', max_depth=None, confidence_threshold=0.5):
        super(RFConfidencePredictor, self).__init__()
        self._BUY_SIGNAL = buy_signal
        self._SELL_SIGNAL = sell_signal
        self._KEEP_SIGNAL = keep_signal
        self._importances = np.zeros(22)
        self._n_estimators = n_estimators
        self._max_features = max_features
        self._max_depth = max_depth

        self._classifier = RandomForestClassifier(n_estimators=self._n_estimators,
                                                  max_features=self._max_features,
                                                  max_depth=self._max_depth,
                                                  random_state=42)
        self._trained_samples = 0
        self._confidences = []
        self._confidence_threshold = confidence_threshold

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

        # Look at confidence of the one class only, as in a binary classification scenario the scores will add up to 1.
        # If sell_signal is greater than buy_signal, this confidence score
        # is the probability for a sell operation and vice versa.
        confidence = self._classifier.predict_proba([todays_features])[0, 1]
        self._confidences.append(confidence)
        if confidence > self._confidence_threshold:
            signal = self._SELL_SIGNAL if self._SELL_SIGNAL > self._BUY_SIGNAL else self._BUY_SIGNAL
        elif 1 - confidence > self._confidence_threshold:
            signal = self._BUY_SIGNAL if self._SELL_SIGNAL > self._BUY_SIGNAL else self._SELL_SIGNAL
        else:
            signal = self._KEEP_SIGNAL
        return signal

    def tune(self, stock_data, symbols, num_features, measure='f1', trading_frequency=10, training_years=3, trading_days_per_year=246):
        n_est_space = [10, 25, 50, 100]
        sqrt_feat = sqrt(num_features)
        max_f_space = [round(sqrt_feat)]
        # max_d_space = [2, 3, 5, 9, None]
        max_d_space = [None]
        conf_threshold_space = [0.55, 0.6, 0.65, 0.7]
        accuracies = []
        f_scores = []
        params = []
        for conf_threshold in conf_threshold_space:
            for n_estimators in n_est_space:
                for max_features in max_f_space:
                    for max_depth in max_d_space:
                        accs = []
                        fs = []
                        for stock in symbols:
                            self._n_estimators = n_estimators
                            self._max_features = max_features
                            self._max_depth = max_depth
                            self._trained_samples = 0
                            self._confidence_threshold = conf_threshold
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
                                preds.append(self.predict(X.iloc[:last_trading_day].values, y.iloc[:last_trading_day].values, X.iloc[trading_day],
                                                          training_years, trading_days_per_year))
                                last_trading_day = trading_day
                            preds = pd.Series(preds, name='Predictions')

                            accuracy = metrics.accuracy_score(y.iloc[first_trading_day:], preds)
                            accs.append(accuracy)

                            non_keep_signals = preds != self._KEEP_SIGNAL
                            f_score = metrics.f1_score(y.iloc[first_trading_day:].values[non_keep_signals], preds[non_keep_signals])
                            fs.append(f_score)

                        mean_acc = np.array(accs).mean()
                        accuracies.append(mean_acc)

                        mean_f_score = np.array(fs).mean()
                        f_scores.append(mean_f_score)

                        params.append((n_estimators, max_features, max_depth, conf_threshold))
                        print('\nRF_c  n_estimators=%i max_features=%i max_depth=%s conf_threshold=%s' % (n_estimators, max_features, max_depth, conf_threshold))
                        print('RF_c    Accuracy: %.3f' % mean_acc)
                        print('RF_c    F1 Score: %.3f' % mean_f_score)
        accuracies = np.array(accuracies)
        f_scores = np.array(f_scores)
        print('\nRF_c FINAL RESULTS')
        if measure == 'accuracy':
            (n_estimators, max_features, max_depth, conf_threshold) = params[accuracies.argmax()]
            print('RF_c  Best result: accuracy %.3f (n_estimators=%i max_features=%i max_depth=%s conf_threshold=%s)' % (accuracies.max(), n_estimators, max_features, max_depth, conf_threshold))
        elif measure == 'f1':
            (n_estimators, max_features, max_depth, conf_threshold) = params[f_scores.argmax()]
            print('RF_c  Best result: f score %.3f (n_estimators=%i max_features=%i max_depth=%s conf_threshold=%s)' % (f_scores.max(), n_estimators, max_features, max_depth, conf_threshold))
        else:
            raise NotImplementedError
        return [n_estimators, max_features, max_depth]
