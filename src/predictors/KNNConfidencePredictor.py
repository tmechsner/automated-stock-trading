import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

from predictors.AbstractPredictor import AbstractPredictor


class KNNConfidencePredictor(AbstractPredictor):
    def __init__(self, buy_signal, sell_signal, keep_signal, confidence_threshold=0.5, k=5, weights='uniform'):
        super(KNNConfidencePredictor, self).__init__()
        self._k = k
        self._weights = weights
        self._classifier = KNeighborsClassifier(n_neighbors=self._k, weights=self._weights)
        self._trained_samples = 0
        self._confidences = []
        self._confidence_threshold = confidence_threshold
        self._BUY_SIGNAL = buy_signal
        self._SELL_SIGNAL = sell_signal
        self._KEEP_SIGNAL = keep_signal

    def predict(self, X, y, todays_features, training_years, trading_days_per_year) -> int:
        new_samples_since_training = len(y[self._trained_samples:])
        if new_samples_since_training >= training_years * trading_days_per_year:
            self._classifier.fit(X, y)
            self._trained_samples = len(y)

        # predict_proba returns a probability score for each class.
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
        k_min = 5
        k_max = 21
        param_space = range(k_min, k_max + 1)[::2]
        conf_threshold_space = [0.55, 0.6, 0.65, 0.7]
        accuracies = []
        f_scores = []
        params = []
        for weights in ['uniform', 'distance']:
            for conf_threshold in conf_threshold_space:
                for k in param_space:
                    accs = []
                    fs = []
                    for stock in symbols:
                        self._k = k
                        self._trained_samples = 0
                        self._confidence_threshold = conf_threshold
                        self._classifier = KNeighborsClassifier(n_neighbors=self._k, weights=weights)

                        preds = []

                        X = stock_data[stock][['I{}'.format(x) for x in range(1, num_features + 1)]]
                        y = stock_data[stock]['target']
                        first_trading_day = training_years * trading_days_per_year + trading_frequency
                        last_trading_day = first_trading_day - trading_frequency
                        for trading_day in range(first_trading_day, len(y)):
                            preds.append(self.predict(X.iloc[:last_trading_day].values, y.iloc[:last_trading_day].values,
                                                      X.iloc[trading_day], training_years, trading_days_per_year))
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

                    params.append((k, weights, conf_threshold))
                    print('\nk-NN_c  %s, k=%i, conf_threshold=%s' % (weights, k, conf_threshold))
                    print('k-NN_c    Accuracy: %.3f' % mean_acc)
                    print('k-NN_c    F1 Score: %.3f' % mean_f_score)
        accuracies = np.array(accuracies)
        f_scores = np.array(f_scores)
        print('\nk-NN_c FINAL RESULTS')
        if measure == 'accuracy':
            (k, weights, conf_threshold) = params[accuracies.argmax()]
            print('k-NN_c  Best result: accuracy %.3f (%s, k=%i, conf_threshold=%s)' % (accuracies.max(), weights, k, conf_threshold))
        elif measure == 'f1':
            (k, weights, conf_threshold) = params[f_scores.argmax()]
            print('k-NN_c  Best result: f score %.3f (%s, k=%i, conf_threshold=%s)' % (f_scores.max(), weights, k, conf_threshold))
        else:
            raise NotImplementedError
        return [k, weights]
