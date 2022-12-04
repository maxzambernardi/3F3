from main import ksdensity
import numpy as np
import matplotlib.pyplot as plt
plt.rc('legend',fontsize=10)
plt.style.use("ggplot")


# define probability density function
def pdf(x):
    y = np.exp(-x)
    return y

# define inverse cdf
def inverse_cdf(x):
    y = np.log(1 / (1 - x))
    return y


# generate vector of 1000 uniformly distributed numbers
vector_uniform = np.random.rand(1000)

# generate kernel function
ks_density_function = ksdensity(inverse_cdf(vector_uniform), width=0.2)

# generate pdf
x_values_pdf = np.linspace(0, 5, 100)
y_values_pdf = pdf(x_values_pdf)


# plots
fig, ax = plt.subplots(1, figsize=(8,8))

# plot pdf vs uniformly distributed numbers transformed through inverse cdf as a histogram
ax.plot(x_values_pdf, y_values_pdf, label="Exact PDF")
ax.hist(inverse_cdf(vector_uniform), bins=50, density=True, label="Histogram Data")
ax.set_xlabel("x", fontname='Times New Roman')
ax.set_ylabel("PDF", fontname='Times New Roman')

# plot pdf vs uniformly distributed numbers transformed through inverse cdf as a histogram
ax.plot(x_values_pdf, ks_density_function(x_values_pdf), label="Kernel Estimation")
ax.legend()
plt.show()
