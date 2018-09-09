import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from predictors.AbstractPredictor import AbstractPredictor


class ETPredictor(AbstractPredictor):
    def __init__(self):
        super(ETPredictor, self).__init__()
        self._importances = np.zeros(22)

    def feature_relevance(self):
        indices = np.argsort(self._importances)[::-1]

        # Print the feature ranking
        print("ET Feature ranking:")

        for f in range(len(indices)):
            print("%d. feature %d (%f)" % (f + 1, indices[f], self._importances[indices[f]]))

    def predict(self, X, y, training_years, trading_days_per_year) -> pd.Series:
        test_years = 1

        # No trading in the first trading_years years -> set predictions to zero
        all_preds = pd.Series(np.zeros(trading_days_per_year * training_years), name='Signal')

        max_trade_years = (len(stock_data) // trading_days_per_year) + 1 - training_years
        days_left = 0
        for current_year in range(training_years, training_years + max_trade_years):
            start_train = trading_days_per_year * (current_year - training_years)
            end_train = trading_days_per_year * current_year
            start_test = end_train
            end_test = trading_days_per_year * (current_year + test_years)

            if end_train >= len(stock_data):
                days_left = end_test - len(stock_data)
                break

            if end_test >= len(stock_data):
                end_test = len(stock_data)

            train_data = stock_data.iloc[start_train:end_train]
            test_data = stock_data.iloc[start_test:end_test]

            X_train = train_data[['I{}'.format(x) for x in range(1, num_features + 1)]]
            y_train = train_data['target']

            X_test = test_data[['I{}'.format(x) for x in range(1, num_features + 1)]]
            y_test = test_data['target']

            clf = ExtraTreesClassifier()
            clf.fit(X_train, y_train)

            importances = clf.feature_importances_
            indices = np.argsort(importances)[::-1]
            for f in range(len(indices)):
                self._importances[indices[f]] += importances[indices[f]]

            preds = pd.Series(clf.predict(X_test), name='Signal')
            all_preds = pd.concat([all_preds, preds])

        if days_left > 0:
            to_fill = range(trading_days_per_year * (training_years + max_trade_years + test_years) - days_left,
                            len(stock_data['target']))
            print(to_fill)
            for _ in to_fill:
                all_preds.append([1])
        all_preds.index = stock_data.index

        return all_preds

    def tune(self, stock_data, symbols, num_features, training_years=3, trading_days_per_year=246):

        pass
        # i = 0
        # k_min = 6
        # k_max = 7
        # results = np.zeros(2 * (k_max - k_min + 1))
        # params = []
        # for weights in ['uniform', 'distance']:
        #     for k in range(k_min, k_max + 1):
        #         params.append((k, weights))
        #         accs = []
        #         for stock in symbols:
        #             classifier = SAMKNN(n_neighbors=k,
        #                                 maxSize=5000,
        #                                 knnWeights=weights,
        #                                 recalculateSTMError=False,
        #                                 useLTM=True)
        #
        #             stock_data[stock] = handle_outliers(stock_data[stock])
        #
        #             X = stock_data[stock][['I{}'.format(x) for x in range(1, num_features + 1)]]
        #             y = stock_data[stock]['target']
        #
        #             predictedLabels, complexity, complexityNumParameterMetric = classifier.trainOnline(np.array(X), np.array(y), np.unique(y))
        #
        #             preds = pd.Series(predictedLabels, name='Signal')
        #             preds[0:trading_days_per_year * training_years] = 0  # for fairness: Don't trade in knn's training years
        #             # preds.index = X.index
        #             accuracy = metrics.accuracy_score(stock_data[stock]['target'].iloc[trading_days_per_year * training_years:],
        #                                               preds.iloc[trading_days_per_year * training_years:])
        #             accs.append(accuracy)
        #         mean_acc = np.array(accs).mean()
        #         results[i] = mean_acc
        #         i += 1
        #         print('   Accuracy %s, k=%i : %.2f' % (weights, k, mean_acc))
        # (k, weights) = params[results.argmax()]
        # print(' Best result: accuracy %.2f (k=%i, %s)' % (results.max(), k, weights))
        # self._k = k
        # self._weights = weights