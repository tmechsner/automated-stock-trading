from scipy.stats.morestats import asarray, compress, find_repeats, warnings, WilcoxonResult
from scipy.stats import stats
from scipy.stats import distributions
from scipy.stats import wilcoxon
import numpy as np
from numpy import sqrt


def wilcoxon_one_sided(x, y=None, zero_method="wilcox", correction=False):
    """
    Calculate the one-tailed Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        The first set of measurements.
    y : array_like, optional
        The second set of measurements.  If `y` is not given, then the `x`
        array is considered to be the differences between the two sets of
        measurements.
    zero_method : string, {"pratt", "wilcox", "zsplit"}, optional
        "pratt":
            Pratt treatment: includes zero-differences in the ranking process
            (more conservative)
        "wilcox":
            Wilcox treatment: discards all zero-differences
        "zsplit":
            Zero rank split: just like Pratt, but spliting the zero rank
            between positive and negative ones
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.

    Returns
    -------
    statistic : float
        The sum of the ranks of the differences above or below zero, whichever
        is smaller.
    pvalue : float
        The one-sided p-value for the test (null hypothesis: x <= y)

    Notes
    -----
    Because the normal approximation is used for the calculations, the
    samples used should be large.  A typical rule is to require that
    n > 20.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

    """

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' "
                         "or 'pratt' or 'zsplit'")

    if y is None:
        d = asarray(x)
    else:
        x, y = map(asarray, (x, y))
        if len(x) != len(y):
            raise ValueError('Unequal N in wilcoxon.  Aborting.')
        d = x - y

    if zero_method == "wilcox":
        # Keep all non-zero differences
        d = compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        warnings.warn("Warning: sample size too small for normal approximation.")

    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0)
    r_minus = np.sum((d < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    T = r_minus
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = r[d != 0]

    replist, repnum = find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = sqrt(se / 24)
    correction = 0.5 * int(bool(correction)) * np.sign(T - mn)
    z = (T - mn - correction) / se
    prob = distributions.norm.cdf(z)

    return WilcoxonResult(T, prob)


def wilcoxon_test(a, b, alpha, a_name, b_name):
    p_two_sided = wilcoxon(a, b).pvalue
    p_one_sided = wilcoxon_one_sided(a, b).pvalue

    print("\nComparing %s and %s" % (a_name, b_name))
    if p_two_sided < alpha:
        print("The two samples differ significantly. (p=%.4f)" % p_two_sided)
        if p_one_sided < alpha:
            print("The first sample ('%s') is greater than the second('%s'). (p=%.4f)" % (a_name, b_name, p_one_sided))
        else:
            print("The first sample ('%s') is lower than the second ('%s'). (p=%.4f)" % (a_name, b_name, p_one_sided))
    else:
        print("The two samples do not differ significantly. (p=%.4f)" % p_two_sided)


def wilcoxon_test_test(mean_a, var_a, mean_b, var_b, n, alpha):
    np.random.seed(42)
    a = np.random.normal(mean_a, var_a, n)
    b = np.random.normal(mean_b, var_b, n)

    wilcoxon_test(a, b, alpha, 'a', 'b')


if __name__ == '__main__':
    wilcoxon_test_test(mean_a=9, var_a=2,
                     mean_b=10, var_b=2,
                     n=100, alpha=0.01)
