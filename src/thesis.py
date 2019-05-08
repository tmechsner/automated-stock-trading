import csv
import datetime
import multiprocessing

from StockTrader import StockTrader
from predictors.KNNPredictor import KNNPredictor
from predictors.SAMKNNPredictor import SAMKNNPredictor
from predictors.KNNConfidencePredictor import KNNConfidencePredictor
from predictors.BuyAndHoldPredictor import BuyAndHoldPredictor
from predictors.RFPredictor import RFPredictor
from predictors.ARFPredictor import ARFPredictor
from predictors.RFConfidencePredictor import RFConfidencePredictor
from predictors.MajorityPredictor import MajorityPredictor

from stats import wilcoxon_one_sided, wilcoxon_test
import numpy as np


def to_csv(data, name):
    csv_file = "results/%s %s.csv" % (name, datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=';')
            writer.writerow(data.keys())
            writer.writerows(zip(*list(data.values())))
    except IOError:
        print("I/O error")


def from_csv(name, date):
    result = {}
    csv_file = "results/%s %s.csv" % (name, date)
    try:
        with open(csv_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            firstrow = True
            keys = None
            for row in reader:
                if firstrow:
                    result = {k: [] for k in row}
                    keys = list(result.keys())
                    firstrow = False
                else:
                    for i, v in enumerate(row):
                        result[keys[i]].append(np.float64(v))
        return result
    except IOError:
        print("I/O error")


def tune(simulation):
    pool = multiprocessing.Pool(processes=6)
    pool.map(simulation.tune_predictor, [
        KNNPredictor(),
        SAMKNNPredictor(),
        KNNConfidencePredictor(simulation.BUY_SIGNAL, simulation.SELL_SIGNAL, simulation.KEEP_SIGNAL),
        RFPredictor(),
        ARFPredictor(),
        RFConfidencePredictor(simulation.BUY_SIGNAL, simulation.SELL_SIGNAL, simulation.KEEP_SIGNAL)
    ])
    pool.close()


def simulate(simulation):
    force_recalculate = True

    simulation.add_predictor(BuyAndHoldPredictor, [simulation.BUY_SIGNAL], name='B&H', force_recalculate=force_recalculate)
    simulation.add_predictor(MajorityPredictor, [], name='Maj', force_recalculate=force_recalculate)
    simulation.add_predictor(KNNPredictor, [21, 'uniform'], 'kNN', force_recalculate=force_recalculate)
    simulation.add_predictor(KNNConfidencePredictor, [simulation.BUY_SIGNAL, simulation.SELL_SIGNAL, simulation.KEEP_SIGNAL, 0.7, 21, 'uniform'], 'kNN_conf_07', force_recalculate=force_recalculate)
    simulation.add_predictor(KNNConfidencePredictor, [simulation.BUY_SIGNAL, simulation.SELL_SIGNAL, simulation.KEEP_SIGNAL, 0.65, 21, 'uniform'], 'kNN_conf_065', force_recalculate=force_recalculate)
    simulation.add_predictor(KNNConfidencePredictor, [simulation.BUY_SIGNAL, simulation.SELL_SIGNAL, simulation.KEEP_SIGNAL, 0.6, 21, 'uniform'], 'kNN_conf_06', force_recalculate=force_recalculate)
    simulation.add_predictor(SAMKNNPredictor, [5, 'distance'], name='SAM', force_recalculate=force_recalculate)
    simulation.add_predictor(SAMKNNPredictor, [10, 'distance', 11], name='SAM_opt', force_recalculate=force_recalculate)
    simulation.add_predictor(RFPredictor, [10, 3, None], 'RF', force_recalculate=force_recalculate)
    simulation.add_predictor(RFConfidencePredictor, [simulation.BUY_SIGNAL, simulation.SELL_SIGNAL, simulation.KEEP_SIGNAL, 100, 3, None, 0.7], 'RF_conf_07', force_recalculate=force_recalculate)
    simulation.add_predictor(RFConfidencePredictor, [simulation.BUY_SIGNAL, simulation.SELL_SIGNAL, simulation.KEEP_SIGNAL, 100, 3, None, 0.65], 'RF_conf_065', force_recalculate=force_recalculate)
    simulation.add_predictor(RFConfidencePredictor, [simulation.BUY_SIGNAL, simulation.SELL_SIGNAL, simulation.KEEP_SIGNAL, 100, 3, None, 0.6], 'RF_conf_06', force_recalculate=force_recalculate)
    simulation.add_predictor(ARFPredictor, [100, 3, 20], 'ARF', force_recalculate=force_recalculate)

    simulation.run_backtest()
    f_scores, balances = simulation.evaluate('B&H')
    to_csv(f_scores, 'f_scores')
    to_csv(balances, 'balances')

    compare_results(balances, f_scores, alpha=0.05)

    simulation.plot_single_value_strategy_balances()


def compare_persisted(date, alpha):
    f_scores = from_csv('f_scores', date)
    balances = from_csv('balances', date)
    compare_results(balances, f_scores, alpha)


def compare_results(balances, f_scores, alpha=0.05):
    print("\nF1 Scores vs Baseline")
    wilcoxon_test(f_scores['kNN'], f_scores['Maj'], alpha, 'kNN', 'Maj')
    wilcoxon_test(f_scores['kNN_conf_07'], f_scores['Maj'], alpha, 'kNN_conf_07', 'Maj')
    wilcoxon_test(f_scores['kNN_conf_065'], f_scores['Maj'], alpha, 'kNN_conf_065', 'Maj')
    wilcoxon_test(f_scores['kNN_conf_06'], f_scores['Maj'], alpha, 'kNN_conf_06', 'Maj')
    wilcoxon_test(f_scores['SAM'], f_scores['Maj'], alpha, 'SAM', 'Maj')
    wilcoxon_test(f_scores['SAM_opt'], f_scores['Maj'], alpha, 'SAM_opt', 'Maj')
    wilcoxon_test(f_scores['RF'], f_scores['Maj'], alpha, 'RF', 'Maj')
    wilcoxon_test(f_scores['RF_conf_07'], f_scores['Maj'], alpha, 'RF_conf_07', 'Maj')
    wilcoxon_test(f_scores['RF_conf_065'], f_scores['Maj'], alpha, 'RF_conf_065', 'Maj')
    wilcoxon_test(f_scores['RF_conf_06'], f_scores['Maj'], alpha, 'RF_conf_06', 'Maj')
    wilcoxon_test(f_scores['ARF'], f_scores['Maj'], alpha, 'ARF', 'Maj')
    print("\nF1 Scores vs Base Clf")
    wilcoxon_test(f_scores['kNN_conf_07'], f_scores['kNN'], alpha, 'kNN_conf_07', 'kNN')
    wilcoxon_test(f_scores['kNN_conf_065'], f_scores['kNN'], alpha, 'kNN_conf_065', 'kNN')
    wilcoxon_test(f_scores['kNN_conf_06'], f_scores['kNN'], alpha, 'kNN_conf_06', 'kNN')
    wilcoxon_test(f_scores['SAM'], f_scores['kNN'], alpha, 'SAM', 'kNN')
    wilcoxon_test(f_scores['SAM_opt'], f_scores['kNN'], alpha, 'SAM_opt', 'kNN')
    wilcoxon_test(f_scores['RF_conf_07'], f_scores['RF'], alpha, 'RF_conf_07', 'RF')
    wilcoxon_test(f_scores['RF_conf_065'], f_scores['RF'], alpha, 'RF_conf_065', 'RF')
    wilcoxon_test(f_scores['RF_conf_06'], f_scores['RF'], alpha, 'RF_conf_06', 'RF')
    wilcoxon_test(f_scores['ARF'], f_scores['RF'], alpha, 'ARF', 'RF')
    print("\nBalances vs Baseline")
    wilcoxon_test(balances['Maj'], balances['B&H'], alpha, 'Maj', 'B&H')
    wilcoxon_test(balances['kNN'], balances['B&H'], alpha, 'kNN', 'B&H')
    wilcoxon_test(balances['kNN_conf_07'], balances['B&H'], alpha, 'kNN_conf_007', 'B&H')
    wilcoxon_test(balances['kNN_conf_065'], balances['B&H'], alpha, 'kNN_conf_065', 'B&H')
    wilcoxon_test(balances['kNN_conf_06'], balances['B&H'], alpha, 'kNN_conf_06', 'B&H')
    wilcoxon_test(balances['SAM'], balances['B&H'], alpha, 'SAM', 'B&H')
    wilcoxon_test(balances['SAM_opt'], balances['B&H'], alpha, 'SAM_opt', 'B&H')
    wilcoxon_test(balances['RF'], balances['B&H'], alpha, 'RF', 'B&H')
    wilcoxon_test(balances['RF_conf_07'], balances['B&H'], alpha, 'RF_conf_07', 'B&H')
    wilcoxon_test(balances['RF_conf_065'], balances['B&H'], alpha, 'RF_conf_065', 'B&H')
    wilcoxon_test(balances['RF_conf_06'], balances['B&H'], alpha, 'RF_conf_06', 'B&H')
    wilcoxon_test(balances['ARF'], balances['B&H'], alpha, 'ARF', 'B&H')
    print("\nBalances vs Base Clf")
    wilcoxon_test(balances['kNN_conf_07'], balances['kNN'], alpha, 'kNN_conf_07', 'kNN')
    wilcoxon_test(balances['kNN_conf_065'], balances['kNN'], alpha, 'kNN_conf_065', 'kNN')
    wilcoxon_test(balances['kNN_conf_06'], balances['kNN'], alpha, 'kNN_conf_06', 'kNN')
    wilcoxon_test(balances['SAM'], balances['kNN'], alpha, 'SAM', 'kNN')
    wilcoxon_test(balances['SAM_opt'], balances['kNN'], alpha, 'SAM_opt', 'kNN')
    wilcoxon_test(balances['RF_conf_07'], balances['RF'], alpha, 'RF_conf_07', 'RF')
    wilcoxon_test(balances['RF_conf_065'], balances['RF'], alpha, 'RF_conf_065', 'RF')
    wilcoxon_test(balances['RF_conf_06'], balances['RF'], alpha, 'RF_conf_06', 'RF')
    wilcoxon_test(balances['ARF'], balances['RF'], alpha, 'ARF', 'RF')


ST_TUNE = 0
ST_SIMULATE = 1


def run(tune_or_simulate):
    """
    Set up the StockTrader and run a simulation on it or tune predictor parameters on it.
    :param tune_or_simulate: Determines whether to run simulation or tuning (ST_SIMULATE or ST_TUNE)
    :return:
    """

    start_date = datetime.date(2002, 6, 3)
    end_date = datetime.date(2017, 5, 31)
    simulation = StockTrader(
        start_date=start_date,
        end_date=end_date,
        dataset='sp500_%s' % ('simulate' if tune_or_simulate else 'tune'),
        trading_frequency=10,
        trans_cost_fixed=99,
        trans_cost_percentual=0.0,
        min_trans=1,
        training_years=3,
        trading_days_per_year=246
    )

    simulation.print_feature_stats()

    if tune_or_simulate == ST_SIMULATE:
        simulate(simulation)
    elif tune_or_simulate == ST_TUNE:
        tune(simulation)


if __name__ == '__main__':

    run_new_simulation = True

    if run_new_simulation:
        run(tune_or_simulate=ST_SIMULATE)
    else:
        compare_persisted('2018-09-08 05-06-56', 0.05)
