import numpy as np
import pandas as pd
from sklearn import metrics

from SAMkNN.SAMKNN.SAMKNNClassifier import SAMKNN
from predictors.AbstractPredictor import AbstractPredictor


class SAMKNNPredictor(AbstractPredictor):
    def __init__(self, k=5, weights='distance', l_min=50):
        super(SAMKNNPredictor, self).__init__()
        self._k = k
        self._weights = weights
        self._classifier = SAMKNN(n_neighbors=self._k,
                                  minSTMSize=l_min,
                                  maxSize=5000,
                                  knnWeights=self._weights,
                                  recalculateSTMError=False,
                                  useLTM=True)
        self._trained_samples = 0

    def predict(self, X, y, todays_features, training_years, trading_days_per_year) -> int:
        self._classifier.trainOnline(X[self._trained_samples:], y[self._trained_samples:], np.unique(y))
        self._trained_samples += len(y[self._trained_samples:])

        return self._classifier.predict([todays_features])[-1]

    def tune(self, stock_data, symbols, num_features, measure='f1', trading_frequency=10, training_years=3, trading_days_per_year=246):
        k_min = 5
        k_max = 21
        param_space = range(k_min, k_max + 1)[::2]
        min_stm_space = [10, 15, 20]
        accuracies = []
        f_scores = []
        params = []
        for weights in ['uniform', 'distance']:
            for min_stm in min_stm_space:
                for k in param_space:
                    if k > (min_stm + 10):
                        continue
                    accs = []
                    fs = []
                    for stock in symbols:
                        classifier = SAMKNN(n_neighbors=k,
                                            minSTMSize=min_stm,
                                            maxSize=5000,
                                            knnWeights=weights,
                                            recalculateSTMError=False,
                                            useLTM=True)

                        X = stock_data[stock][['I{}'.format(x) for x in range(1, num_features + 1)]]
                        y = stock_data[stock]['target']

                        predictedLabels, complexity, complexityNumParameterMetric = classifier.trainOnline(X.values, y.values, np.unique(y))

                        preds = pd.Series(predictedLabels, name='Signal')
                        preds.index = y.index

                        if len(np.unique(preds[1:])) > 2:
                            print("SAM output contained non-existent class 0! %s %i" % (stock, k))
                            continue

                        accuracy = metrics.accuracy_score(y.iloc[1:], preds[1:])
                        accs.append(accuracy)

                        f_score = metrics.f1_score(y.iloc[1:], preds[1:])
                        fs.append(f_score)

                    mean_acc = np.array(accs).mean()
                    accuracies.append(mean_acc)

                    mean_f_score = np.array(fs).mean()
                    f_scores.append(mean_f_score)

                    params.append((k, weights, min_stm))
                    print('\nSAM  %s, k=%i, L_min=%i' % (weights, k, min_stm))
                    print('SAM    Accuracy: %.3f' % mean_acc)
                    print('SAM    F1 Score: %.3f' % mean_f_score)
        accuracies = np.array(accuracies)
        f_scores = np.array(f_scores)
        print('\nSAM FINAL RESULTS')
        if measure == 'accuracy':
            argmax = accuracies.argsort()[-5:]
            for i in argmax:
                (k, weights, min_stm) = params[i]
                print('SAM  Best result: accuracy %.3f (k=%i, %s, L_min=%i)' % (accuracies[i], k, weights, min_stm))
        elif measure == 'f1':
            argmax = f_scores.argsort()[-5:]
            for i in argmax:
                (k, weights, min_stm) = params[i]
                print('SAM  Best result: f score %.3f (k=%i, %s, L_min=%i)' % (f_scores[i], k, weights, min_stm))
        else:
            raise NotImplementedError
