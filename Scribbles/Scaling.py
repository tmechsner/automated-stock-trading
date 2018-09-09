import matplotlib.pyplot as plt
import numpy as np


def scale(x):
    x_abs = np.abs(x)
    return np.sign(x) * (x_abs - np.min(x_abs)) / (np.max(x_abs) - np.min(x_abs))

n = 50

t = np.linspace(0, n, n)

x1 = t + np.exp((np.sin(t/4) - 0.3) * 3) - 10
x2 = scale(x1)

fig = plt.figure()

ax = plt.subplot(211)
ax.plot(t, x1, label='time series 1')
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.legend()

ax = plt.subplot(212)
ax.plot(t, x2)
ax.set_xlabel('lags')
ax.set_ylabel('p-value')

plt.show()