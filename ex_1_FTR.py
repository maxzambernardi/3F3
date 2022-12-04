import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import norm
plt.style.use("ggplot")
plt.rc('legend',fontsize=10)


def plot_gaussian_mean(n_bins):

    fig, ax = plt.subplots(3, figsize=(12,9))

    for i in range(3):
        N = 10**(2+i)
        x_points = np.linspace(-3,3,1000)
        x_samples_uniform = np.random.randn(N)
        ax[i].set_title(f"N={N}", fontname='Times New Roman', fontsize=13)
        (n, bins, patches) = ax[i].hist(x_samples_uniform, bins=n_bins)
        cumulative_prob = norm.cdf(bins)
        p = [cumulative_prob[n]-cumulative_prob[n-1] for n in range(1,len(cumulative_prob))]
        mean = [N*p for p in p]
        var = [N*p*(1-p) for p in p]
        sd = [np.sqrt(var) for var in var]
        three_sd = [3*sd for sd in sd]
        print(len(mean))
        print(len(var))
        print(len(three_sd))
        bins = np.delete(bins, 0)
        ax[i].plot(bins, mean, label="\u03BC")
        ax[i].plot(bins, [mean+three_sd for mean, three_sd in zip(mean,three_sd)] , label="\u03BC +3\u03C3")
        ax[i].plot(bins, [mean-three_sd for mean, three_sd in zip(mean,three_sd)], label="\u03BC -3\u03C3")
        ax[i].legend(loc="upper right")
    plt.show()

plot_gaussian_mean(50)



