import numpy as np
import matplotlib.pyplot as plt
plt.rc('legend',fontsize=10)
plt.style.use("ggplot")



# define f(x)
def f(x, a=5, b=10):
    y = a * x + b
    return y


# define g(x)
def g(x):
    y = x ** 2
    return y


# gaussian function
def n_pdf(x, mu=0., sigma=1.):  # normal pdf
    u = (x - mu) / abs(sigma)
    y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
    y *= np.exp(-u * u / 2)
    return y


# define exponential function
def exp_f(x):
    y = (1 / (np.sqrt(2 * np.pi*x))) * np.exp(-x / 2)
    return y


# generate vector of 1000 Gaussian numbers
vector_gaussian = np.random.randn(1000)

# generate normal distribution
x_values_gaussian = np.linspace(-5, 25, 100)
y_values_gaussian = n_pdf(x_values_gaussian, mu=10, sigma=5)

# generate exponential function
x_values_exp = np.linspace(0.1, 5, 100)
y_values_exp = exp_f(x_values_exp)


# y = ax +b transformation
plt.hist(f(vector_gaussian), bins=30, density=True, label="Transformation Data")
plt.plot(x_values_gaussian, y_values_gaussian, label="PDF")
plt.legend(loc="upper right")
plt.xlabel("x", fontname='Times New Roman')
plt.ylabel("PDF", fontname='Times New Roman')
plt.show()
plt.clf()

# y = x^2 transformation
plt.hist(g(vector_gaussian), bins=40, density=True, label="Transformation Data")
plt.plot(x_values_exp, y_values_exp, label="PDF")
plt.legend(loc="upper right")
plt.xlabel("x", fontname='Times New Roman')
plt.ylabel("PDF", fontname='Times New Roman')
plt.show()
plt.clf()
