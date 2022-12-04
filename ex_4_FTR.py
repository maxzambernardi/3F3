import numpy as np
import matplotlib.pyplot as plt
from main import ksdensity
import random

plt.style.use("ggplot")
plt.rc('legend', fontsize=10)

def f1(x, alpha=1):
    return -alpha*np.absolute(x)

def f2(x, alpha=1):
    return -alpha*(np.absolute(x)**2)

def f3(x, alpha=1):
    return -alpha*(np.absolute(x)**(0.75))

def n_pdf(x, mu=0., sigma=1.):  # normal pdf
    u = (x - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
    y *= np.exp(-u * u / 2)
    return y

# define inverse CDF of p(u)
def inverse_cdf(x, alpha=1):
    y = (2 / alpha ** 2) * np.log(1 / (1 - x))
    return y



# state number of samples
N = 10000

# pick a value for theta (shape of the gamma distribuition)
theta = [0.1, 0.5, 5, 50]
widths = [1.2, 0.9, 0.3, 0.3]

# create a subplot to store results
fig, ax = plt.subplots(2,2, figsize=(14, 12))
coord = [(0,0), (0,1), (1,0), (1,1)]


for index, t, w in zip(coord, theta, widths):

    # generate samples from the gamma distribution
    v = np.random.gamma(shape=t, scale=(1/t), size=N)

    variance_gamma = 1/v

    # sample from a normal distribution
    samples_gamma = np.zeros(N)
    for i in range(N):
        samples_gamma[i] = np.random.normal(scale=np.sqrt(variance_gamma[i]))
    print(np.sqrt(np.var(samples_gamma)))

    # plot the gamma_like distribution
    (n, bins, patches) = ax[index].hist(samples_gamma[abs(samples_gamma)<40], bins=50, label="Histogram data")
    x_values = np.linspace(np.min(samples_gamma[abs(samples_gamma)<40]), np.max(samples_gamma[abs(samples_gamma)<40]), 1000)
    scaling_factor = len(samples_gamma) * (bins[1] - bins[0])
    ks_density_function = ksdensity(samples_gamma, width=w)
    ax[index].plot(x_values, ks_density_function(x_values) * scaling_factor, label="Kernel Estimation")
    ax[index].legend(loc="upper right")
    ax[index].set_title(f"\u03B8={t}",fontname='Times New Roman', fontsize=13)
    ax[index].set_xlabel("x", fontname='Times New Roman')
    ax[index].set_ylabel("N", fontname='Times New Roman')


plt.show()



# investigate exponential characteristic when theta = 1, alpha = 1/sqrt(2)
theta = 1
alpha = np.sqrt(2)

# generate samples from the gamma distribution
v = np.random.gamma(shape=theta, scale=(1/theta), size=N)
variance_gamma = 1/v
# sample from a normal distribution
samples_gamma = np.zeros(N)
for i in range(N):
    samples_gamma[i] = np.random.normal(scale=np.sqrt(variance_gamma[i]))

# generate samples (random variances) from p(u) distribution
x_samples_uniform = np.random.rand(N)
random_variance = inverse_cdf(x_samples_uniform, alpha=alpha)
# generate samples from p(x) (which can be obtained through marginalization)
samples_exponential = np.zeros(N)
for i in range(N):
    samples_exponential[i] = np.random.normal(scale=np.sqrt(random_variance[i]))

fig, ax = plt.subplots(1,2, figsize=(14, 7))
fig_extra, ax_extra = plt.subplots(1)

(n, bins, patches) = ax_extra.hist(samples_gamma[abs(samples_gamma)<10], bins=50, density=True, label="gamma-mixing distribution")
(n1, bins1, pathces1) = ax_extra.hist(samples_exponential[abs(samples_exponential)<10], bins=50, density=True, alpha=0.4, label="laplace distribution")
ax[0].set_xlabel("x", fontname='Times New Roman')
ax[0].set_ylabel("PDF", fontname='Times New Roman')
ax[0].set_title(f"\u03B1={round(alpha,3)}, \u03B8={theta}", fontname='Times New Roman', fontsize=13)

# logarithmic plots
x_values = np.linspace(np.min(samples_gamma[abs(samples_gamma)<10]), np.max(samples_gamma[abs(samples_gamma)<10]), N)
scaling_factor = N*(bins[1]-bins[0])
ks_density_function_gamma = ksdensity(samples_gamma, width=0.3)
ks_density_function_exponential = ksdensity(samples_exponential, width=0.3)
ax[0].plot(x_values, ks_density_function_gamma(x_values), label = "Gamma-mixing kernel")
ax[0].plot(x_values, ks_density_function_exponential(x_values), label= "Exponential-mixing kernel")
ax[0].legend(loc="upper right")

peak_exp = np.max(np.log(ks_density_function_exponential(x_values) * scaling_factor))
ax[1].plot(x_values, np.log(ks_density_function_gamma(x_values) * scaling_factor), label="Gamma-mixing kernel")
ax[1].plot(x_values, np.log(ks_density_function_exponential(x_values) * scaling_factor), label="Exponential-mixing kernel")
ax[1].plot(x_values, f3(x_values, alpha=alpha) + peak_exp, label=f"-{round(alpha,3)}$|x|^{0.75}$")
ax[1].set_ylim(3.5, 1.03 * peak_exp)
ax[1].set_xlim(-6,6)
ax[1].legend(loc="upper right")
ax[1].set_title(f"\u03B1={round(alpha,3)}, \u03B8={theta}", fontname='Times New Roman', fontsize=13)
ax[1].set_xlabel("x", fontname='Times New Roman')
ax[1].set_ylabel("N", fontname='Times New Roman')
plt.show()



# investigate gaussian characteristic when theta is large (10000)
theta = 10000
# generate samples from the gamma distribution
v = np.random.gamma(shape=theta, scale=(1/theta), size=N)
variance_gamma = 1/v
# sample from a normal distribution
samples_gamma = np.zeros(N)
for i in range(N):
    samples_gamma[i] = np.random.normal(scale=np.sqrt(variance_gamma[i]))

# generate gaussian with mean 0 and variance 2
x_gaussian = np.linspace(-6,6,1000)
gaussian_curve = n_pdf(x_gaussian,mu=0,sigma=1)

# kernel estimation
x_values = np.linspace(np.min(samples_gamma), np.max(samples_gamma), N)
ks_density_function_gamma = ksdensity(samples_gamma, width=0.3)

fig, ax = plt.subplots(1,2, figsize=(14,6))

# linear plots
(n, bins, patches) = ax[0].hist(samples_gamma, bins=50, density=True, label="Gamma-mixing distribution")
ax[0].plot(x_gaussian, gaussian_curve, label="Gaussian")
ax[0].plot(x_values, ks_density_function_gamma(x_values), label="Kernel Estimation")
ax[0].legend(loc="upper right")
ax[0].set_xlabel("x", fontname='Times New Roman')
ax[0].set_ylabel("PDF", fontname='Times New Roman')
ax[0].set_title(f"\u03B8={theta}, \u03C3={1} ", fontname='Times New Roman', fontsize=13)


# logarithmic plots
scaling_factor = N*(bins[1]-bins[0])
peak = np.max(np.log(ks_density_function_gamma(x_values) * scaling_factor))
ax[1].plot(x_values, np.log(ks_density_function_gamma(x_values) * scaling_factor), label="Gamma-mixing distribution")
ax[1].plot(x_values, 0.5*f1(x_values) + peak, label=f"-0.5|x|")
ax[1].plot(x_values, 0.5*f2(x_values) + peak, label=f"-0.5$x^{2}$") #investigate where the 0.6 comes from
ax[1].set_ylim(3, 1.03 * peak)
ax[1].legend(loc="upper right")
ax[1].set_title(f"\u03B8={theta}, \u03C3={1}", fontname='Times New Roman', fontsize=13)
ax[1].set_xlabel("x", fontname='Times New Roman')
ax[1].set_ylabel("N", fontname='Times New Roman')
plt.show()