import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from statsmodels.tsa.stattools import grangercausalitytests

n = 50

t = np.linspace(0, n, n)

# x1 = t * 0.3 + np.sin((t - n*0.5)**2 * 0.1)
x1 = t * 0.3 + np.random.normal(0, 0.1, n) # + np.random.normal(0, 2, n) * np.sin((t - n*0.5)**2 * 0.1)
x2 = np.zeros(n)
x2[:-3] = x1[3:]
x3 = np.exp(np.sqrt(t) * 0.5) # + np.random.normal(0, 0.05, n) # + np.random.normal(0, 2, n) * np.sin((t - n*0.5)**2 * 0.1)
x4 = -t * 0.3 + np.random.normal(0, 1, n) + np.random.normal(0, 2, n) * np.sin((t - n*0.5)**2 * 0.1)
x5 = np.random.normal(0, 1, n) + np.random.normal(0, 1, n) * np.sin((t - n*0.2) * 0.1) * 4

useLogDiff = False

if useLogDiff:
    x1 = np.diff(x1)
    x2 = np.diff(x2)
    x3 = np.diff(x3)
    x4 = np.diff(x4)
    x5 = np.diff(x5)

    x1 = np.log(1 + x1 - np.min(x1))
    x2 = np.log(1 + x1 - np.min(x2))
    x3 = np.log(1 + x1 - np.min(x3))
    x4 = np.log(1 + x1 - np.min(x4))
    x5 = np.log(1 + x1 - np.min(x5))
    t = t[1:]
    n -= 1



lm = linear_model.LinearRegression()
model = lm.fit(t.reshape(-1, 1), x2)

fig = plt.figure()

ax = plt.subplot(241)
ax.plot(t, x1, label='time series 1')
ax.plot(t, x2, label='time series 2')
# ax.plot(t, lm.predict(t.reshape(-1, 1)), color='red', label='linear regression on ts2')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend()

lags = 5
mat = np.zeros((n, 2))
mat[:, 0] = x1
mat[:, 1] = x2
gr = grangercausalitytests(mat, lags, verbose=False)
p_values = [gr[x][0]['params_ftest'][1] for x in gr.keys()]

ax = plt.subplot(245)
ax.plot(range(1, lags + 1), list(p_values), label='granger causality')
ax.set_xlabel('lags')
ax.set_ylabel('p-value')




ax = plt.subplot(242)
ax.plot(t, x1)
ax.plot(t, x3)
# ax.plot(t, lm.predict(t.reshape(-1, 1)), color='red', label='linear regression on ts2')
ax.set_xlabel('t')

mat = np.zeros((n, 2))
mat[:, 0] = x1
mat[:, 1] = x3
gr = grangercausalitytests(mat, lags, verbose=False)
p_values = [gr[x][0]['params_ftest'][1] for x in gr.keys()]

ax = plt.subplot(246)
ax.plot(range(1, lags + 1), list(p_values), label='granger causality')
ax.set_xlabel('lags')




ax = plt.subplot(243)
ax.plot(t, x1)
ax.plot(t, x4)
# ax.plot(t, lm.predict(t.reshape(-1, 1)), color='red', label='linear regression on ts2')
ax.set_xlabel('t')

mat = np.zeros((n, 2))
mat[:, 0] = x1
mat[:, 1] = x4
gr = grangercausalitytests(mat, lags, verbose=False)
p_values = [gr[x][0]['params_ftest'][1] for x in gr.keys()]

ax = plt.subplot(247)
ax.plot(range(1, lags + 1), list(p_values), label='granger causality')
ax.set_xlabel('lags')




ax = plt.subplot(244)
ax.plot(t, x1)
ax.plot(t, x5)
# ax.plot(t, lm.predict(t.reshape(-1, 1)), color='red', label='linear regression on ts2')
ax.set_xlabel('t')

mat = np.zeros((n, 2))
mat[:, 0] = x1
mat[:, 1] = x5
gr = grangercausalitytests(mat, lags, verbose=False)
p_values = [gr[x][0]['params_ftest'][1] for x in gr.keys()]

ax = plt.subplot(248)
ax.plot(range(1, lags + 1), list(p_values), label='granger causality')
ax.set_xlabel('lags')

plt.show()
