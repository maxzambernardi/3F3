import numpy as np
import matplotlib.pyplot as plt
from main import ksdensity
import random

plt.style.use("ggplot")
plt.rc('legend', fontsize=10)


N = 10000
theta = [0.1, 0.25, 0.4, 0.5]
widths = [1.2, 1.2, 1.2, 1.2]
x_lim = [100, 100, 100, 100]

# create a subplot to store results
fig, ax = plt.subplots(2,2, figsize=(14, 12))
coord = [(0,0), (0,1), (1,0), (1,1)]



for index, t, w, x in zip(coord, theta, widths, x_lim):
    # generate samples from the gamma distribution
    v = np.random.gamma(shape=t, scale=(1/t), size=N)

    variance_gamma = 1/v

    # sample from a normal distribution
    samples_gamma = np.zeros(N)
    for i in range(N):
        samples_gamma[i] = np.random.normal(scale=np.sqrt(variance_gamma[i]))

    ks_density_function = ksdensity(samples_gamma, width=w)
    x_values = np.linspace(-x, x, 1000)
    x_values_tail = np.linspace(1, x, 1000)
    ax[index].set_ylim(-12, 0)
    ax[index].plot(x_values, np.log(ks_density_function(x_values)),label = "Kernel Density Estimation")
    peak = np.max(np.log(ks_density_function(x_values)))
    decay_rate = -2*(t+0.5)
    ax[index].plot(x_values_tail, np.log([x**(-2*(t+0.5)) for x in x_values_tail])+peak, label = "$x^{-2(\u03B8 +0.5)}$")
    ax[index].set_xlim(0,x_values_tail[-1])
    ax[index].legend(loc="upper right")
    ax[index].set_title(f"\u03B8={t}", fontname='Times New Roman', fontsize=13)
    ax[index].set_xlabel("x", fontname='Times New Roman')
    ax[index].set_ylabel("ln(p(u))", fontname='Times New Roman')

plt.show()