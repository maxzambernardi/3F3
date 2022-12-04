from main import ksdensity
import numpy as np
import matplotlib.pyplot as plt

plt.rc('legend', fontsize=10)
plt.style.use("ggplot")


# define inverse cdf
def inverse_cdf(x):
    y = np.log(1 / (1 - x))
    return y

fig, ax = plt.subplots(2, figsize=(8,13))

def monte_carlo_estimator(N):
    error_log = []
    error_linear = []
    real_mean = 1
    x_values_linear = np.linspace(1000,10**N,100)
    x_values_log = np.logspace(3,N,100)

    for i,k in zip(x_values_log,x_values_linear):
        error_n_log = []
        error_n_linear = []
        for j in range(10):
            # generate vector of uniformly distributed numbers
            vector_uniform_log = np.random.rand(int(i))
            vector_uniform_linear = np.random.rand(int(k))

            # samples from exponential distribution
            exp_samples_log = inverse_cdf(vector_uniform_log)
            exp_samples_linear = inverse_cdf(vector_uniform_linear)

            # estimate mean and variance using Monte Carlo
            mean_log = (1 / len(exp_samples_log)) * sum(exp_samples_log)
            error_n_log.append((real_mean - mean_log) ** 2)

            mean_linear = (1 / len(exp_samples_linear)) * sum(exp_samples_linear)
            error_n_linear.append((real_mean - mean_linear) ** 2)

        error_n_log = np.average(error_n_log)
        error_n_linear = np.average(error_n_linear)
        error_log.append(error_n_log)
        error_linear.append(error_n_linear)


    ax[0].plot(x_values_linear, error_linear, label="Experimental")
    ax[0].plot(x_values_linear, 1/x_values_linear, label="Theoretical")
    ax[0].legend(loc="upper right")
    ax[0].set_xlabel("N samples", fontname='Times New Roman')
    ax[0].set_ylabel("Squared error", fontname='Times New Roman')
    ax[0].set_title("Linear Plot", fontname='Times New Roman', fontsize=13)

    ax[1].loglog(x_values_log, error_log, label="Experimental")
    ax[1].loglog(x_values_log, 1 / x_values_log, label="Theoretical")
    ax[1].legend(loc="upper right")
    ax[1].set_xlabel("N samples", fontname='Times New Roman')
    ax[1].set_ylabel("Squared error", fontname='Times New Roman')
    ax[1].set_title("Logarithmic Plot", fontname='Times New Roman', fontsize=13)


    plt.show()


monte_carlo_estimator(5)

# generate vector of uniformly distributed numbers
vector_uniform_linear = np.random.rand(1000)
# samples from exponential distribution
exp_samples_linear = inverse_cdf(vector_uniform_linear)
#calculate mean and variance
mean = (1/1000)*np.sum(exp_samples_linear)
variance = (1/1000)*np.sum([i**2 for i in exp_samples_linear]) -mean**2
print(mean)
print(variance)