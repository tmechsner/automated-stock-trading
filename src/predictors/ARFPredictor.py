import sys

from math import sqrt
import numpy as np
import pandas as pd
from sklearn import metrics

from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForest
from predictors.AbstractPredictor import AbstractPredictor


class ARFPredictor(AbstractPredictor):
    def __init__(self, n_estimators=10, max_features='auto', grace_period=50):
        super(ARFPredictor, self).__init__()
        self._n_estimators = n_estimators
        self._max_features = max_features
        self._classifier = AdaptiveRandomForest(n_estimators=self._n_estimators,
                                                max_features=self._max_features,
                                                grace_period=grace_period,
                                                random_state=42)
        self._trained_samples = 0

    def predict(self, X, y, todays_features, training_years, trading_days_per_year) -> int:
        self._classifier.partial_fit(X[self._trained_samples:], y[self._trained_samples:], np.unique(y))
        self._trained_samples += len(y[self._trained_samples:])

        prediction = self._classifier.predict([todays_features])
        return prediction[-1]

    def tune(self, stock_data, symbols, num_features, measure='f1', trading_frequency=10, training_years=3, trading_days_per_year=246):
        gp_space = [10, 15, 20]
        n_est_space = [50, 100]
        sqrt_feat = sqrt(num_features)
        max_f_space = [round(sqrt_feat)]
        accuracies = []
        f_scores = []
        params = []
        for gp in gp_space:
            for n_estimators in n_est_space:
                for max_features in max_f_space:
                    accs = []
                    fs = []
                    for stock in symbols:
                        print(stock)
                        self._n_estimators = n_estimators
                        self._max_features = max_features
                        self._trained_samples = 0
                        self._classifier = AdaptiveRandomForest(n_estimators=self._n_estimators,
                                                                max_features=self._max_features,
                                                                grace_period=gp,
                                                                random_state=42)

                        preds = []

                        X = stock_data[stock][['I{}'.format(x) for x in range(1, num_features + 1)]]
                        y = stock_data[stock]['target']
                        first_trading_day = training_years * trading_days_per_year + trading_frequency
                        last_trading_day = first_trading_day - trading_frequency
                        for trading_day in range(first_trading_day, len(y)):
                            sys.stdout.write('\r%i/%i' % (trading_day, len(y)))
                            sys.stdout.flush()
                            preds.append(
                                self.predict(X.iloc[:last_trading_day].values, y.iloc[:last_trading_day].values, X.iloc[trading_day],
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

                    params.append((n_estimators, max_features, gp))
                    print('\nARF  n_estimators=%i max_features=%i gp=%i' % (n_estimators, max_features, gp))
                    print('ARF    Accuracy: %.3f' % mean_acc)
                    print('ARF    F1 Score: %.3f' % mean_f_score)
        accuracies = np.array(accuracies)
        f_scores = np.array(f_scores)
        print('\nARF FINAL RESULTS')
        if measure == 'accuracy':
            (n_estimators, max_features, gp) = params[accuracies.argmax()]
            print('ARF  Best result: accuracy %.3f (n_estimators=%i max_features=%i gp=%i)' % (accuracies.max(), n_estimators, max_features, gp))
        elif measure == 'f1':
            (n_estimators, max_features, gp) = params[f_scores.argmax()]
            print('ARF  Best result: f score %.3f (n_estimators=%i max_features=%i gp=%i)' % (f_scores.max(), n_estimators, max_features, gp))
        else:
            raise NotImplementedError
        return [n_estimators, max_features]
