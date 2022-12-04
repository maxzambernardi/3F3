import numpy as np
import matplotlib.pyplot as plt
from main import ksdensity
import random
from scipy import stats
from scipy import special

plt.style.use("ggplot")
plt.rc('legend', fontsize=10)

def f(x, theta):
    y = (special.gamma(theta+0.5)/special.gamma(theta))*(theta**(theta))*(1/np.sqrt(2*np.pi))*(2/(2*theta+x**2))**(theta+0.5)
    return y

fig, ax = plt.subplots(2,2, figsize=(14, 12))

theta = [1, 5, 10, 100]
coord = [(0,0), (0,1), (1,0), (1,1)]

for index, t in zip(coord, theta):
    # generate samples from the gamma distribution
    v = np.random.gamma(shape=t, scale=(1/t), size=10000)

    variance_gamma = 1/v

    # sample from a normal distribution
    samples_gamma = np.zeros(10000)
    for i in range(10000):
        samples_gamma[i] = np.random.normal(scale=np.sqrt(variance_gamma[i]))

    x_values = np.linspace(-10,10,1000)
    (n, bins, patches) = ax[index].hist(samples_gamma[abs(samples_gamma)<10], bins = 50, density = True, label="Histogram Data")
    ax[index].plot(x_values, f(x_values, t), label="Theoretical PDF")
    ax[index].legend(loc="upper right")
    ax[index].set_title(f"\u03B8={t}", fontname='Times New Roman', fontsize=13)
    ax[index].set_xlabel("x", fontname='Times New Roman')
    ax[index].set_ylabel("PDF", fontname='Times New Roman')

plt.show()






