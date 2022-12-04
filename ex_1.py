import numpy as np
import random
import matplotlib.pyplot as plt
from main import ksdensity
from scipy.stats import uniform
plt.style.use("ggplot")
plt.rc('legend',fontsize=10)


def n_pdf(x, mu=0., sigma=1.):  # normal pdf
    u = (x - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
    y *= np.exp(-u * u / 2)
    return y


# generate vector of 1000 Gaussian numbers
vector_gaussian = np.random.randn(1000)

# generate vector of 1000 numbers from a uniform distribution
vector_uniform = np.random.rand(1000)

# generate normal distribution
x_values_gaussian = np.linspace(-3, 3, 100)
y_values_gaussian = n_pdf(x_values_gaussian)

# generate kernel estimation function for normal distribution and uniform distribution
ks_function_normal = ksdensity(vector_gaussian, width=0.3)
ks_function_uniform = ksdensity(vector_uniform, width=0.2)

# generate uniform distribution
x_values_uniform = np.linspace(-0.5, 1.5, 1000)
y_values_uniform = [1 for i in range(100)]

# create a subplot for results
fig, ax = plt.subplots(2, figsize=(8, 8))

# plot Gaussian curve vs histogram
ax[0].plot(x_values_gaussian, y_values_gaussian, label="Theoretical")
ax[0].hist(vector_gaussian, bins=30, density=True, label="Histogram Data")
ax[0].legend(loc="upper right")
ax[0].set_xlabel("x", fontname='Times New Roman')
ax[0].set_ylabel("PDF", fontname='Times New Roman')

# plot Gaussian curve vs Kernel estimation
ax[1].plot(x_values_gaussian, y_values_gaussian, label="Theoretical")
ax[1].plot(x_values_gaussian, ks_function_normal(x_values_gaussian), label="Kernel Estimation")
ax[1].legend(loc="upper right")
ax[1].set_xlabel("x", fontname='Times New Roman')
ax[1].set_ylabel("PDF", fontname='Times New Roman')

# create a subplot for results
fig, ax = plt.subplots(2, figsize=(8, 8))

# plot uniform curve vs histogram
ax[0].plot(x_values_uniform, uniform.pdf(x_values_uniform), label="Theoretical")
ax[0].hist(vector_uniform, bins=20, density=True, label="Histogram Data")
ax[0].legend(loc="upper right")
ax[0].set_xlabel("x", fontname='Times New Roman')
ax[0].set_ylabel("PDF", fontname='Times New Roman')

# plot uniform curve vs Kernel estimation
ax[1].plot(x_values_uniform, uniform.pdf(x_values_uniform), label="Theoretical")
ax[1].plot(x_values_uniform, ks_function_uniform(x_values_uniform), label="Kernel Estimation")
ax[1].legend(loc="upper right")
ax[1].set_xlabel("x", fontname='Times New Roman')
ax[1].set_ylabel("PDF", fontname='Times New Roman')

# probability of each sample to be in one bin
p = 1/30

# create subplots for results
fig, ax = plt.subplots(3, figsize=(12,8))

for i in range(3):
    N = 10**(2+i)
    mean = N*p
    var = N*p*(1-p)
    sd = np.sqrt(var)
    x_points = np.linspace(0,1,1000)
    x_samples_uniform = np.random.rand(N)
    ax[i].set_title(f"N={N}", fontname='Times New Roman', fontsize=13)
    ax[i].hist(x_samples_uniform, bins=30)
    ax[i].plot(x_points, [mean for i in range(len(x_points))], label="\u03BC")
    ax[i].plot(x_points, [(mean+3*sd) for i in range(len(x_points))], label="\u03BC +3\u03C3")
    ax[i].plot(x_points, [(mean - 3 * sd) for i in range(len(x_points))], label="\u03BC -3\u03C3")
    ax[i].legend(loc="upper right")

plt.show()

plt.show()
