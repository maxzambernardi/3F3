import numpy as np
import matplotlib.pyplot as plt
from main import ksdensity
plt.style.use("ggplot")
plt.rc('legend',fontsize=10)

alpha = 0.1

# define p(u)
def p(u, alpha=1):
    y = ((alpha ** 2) / 2) * (np.exp(-((alpha ** 2) * u) / 2))
    return y


# define inverse CDF of p(u)
def inverse_cdf(x, alpha=1):
    y = (2 / alpha ** 2) * np.log(1 / (1 - x))
    return y


# gaussian function
def n_pdf(x, mu=0., sigma=1.):  # normal pdf
    u = (x - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
    y *= np.exp(-u * u / 2)
    return y

def f1(x, alpha=1):
    return -alpha*np.absolute(x)

def f2(x, alpha=1):
    return -alpha*(x**2)


alpha = [0.1, 0.5, 1, 1.5]
fig, ax = plt.subplots(2,2, figsize=(14, 12))
fig1, ax1 = plt.subplots(2,2, figsize=(14, 12))
coord = [(0,0), (0,1), (1,0), (1,1)]
for index, a in zip(coord, alpha):
    # generate samples (random variances) from p(u) distribution
    x_samples_uniform = np.random.rand(10000)
    random_variance = inverse_cdf(x_samples_uniform, alpha=a)

    # generate samples from p(x) (which can be obtained through marginalization)
    samples = np.zeros(10000)
    for i in range(len(random_variance)):
        samples[i] = np.random.normal(scale=np.sqrt(random_variance[i]))

    # histogram and kernel density estimation of the data
    left = -3*np.std(samples)
    right = 3*np.std(samples)
    (n, bins, patches) = ax[index].hist(samples, bins=50, label="Histogram Data")
    x_values = np.linspace(np.min(samples), np.max(samples), 1000)
    scaling_factor = len(samples) * (bins[1] - bins[0])
    ks_density_function = ksdensity(samples, width=1 / (3 * a))  # tune the width of the kernel based on alpha
    ax[index].set_xlim(left, right)
    ax[index].plot(x_values, ks_density_function(x_values) * scaling_factor, label="Kernel Estimation")
    ax[index].legend(loc="upper right")
    ax[index].set_title(f"\u03B1={a}",fontname='Times New Roman', fontsize=13)
    ax[index].set_xlabel("x", fontname='Times New Roman')
    ax[index].set_ylabel("N", fontname='Times New Roman')

    #compare the shape of the kernel density estimation with the shape of a gaussian with the same variance
    # normalize them for comparison purposes
    data_std = np.std(samples)
    ax[index].plot(x_values, n_pdf(x_values,sigma=data_std)*scaling_factor, label="Exact Gaussian")
    ax[index].legend(loc="upper right")

    #logarithmic plots
    peak = np.max(np.log(ks_density_function(x_values)* scaling_factor))
    ax1[index].plot(x_values, np.log(ks_density_function(x_values)* scaling_factor), label="Kernel Estimation")
    ax1[index].plot(x_values, f1(x_values, alpha=a)+peak, label=f"-{a}|x|")
    ax1[index].plot(x_values, f2(x_values, alpha=a)+peak, label=f"-{a}$x^{2}$")
    ax1[index].set_xlim(left, right)
    ax1[index].set_ylim(0, 1.03*peak)
    ax1[index].legend(loc="lower right")
    ax1[index].set_title(f"\u03B1={a}", fontname='Times New Roman', fontsize=13)
    ax1[index].set_xlabel("x", fontname='Times New Roman')
    ax1[index].set_ylabel("N", fontname='Times New Roman')

plt.show()
