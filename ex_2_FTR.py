import numpy as np
import matplotlib.pyplot as plt
import random
plt.rc('legend',fontsize=10)
plt.style.use("ggplot")

def f(x):
    y = (1/np.pi)*(1/np.sqrt(1-x**2))
    return y


# generate vector of 1000 uniform numbers
vector_uniform = np.random.uniform(0,2*np.pi,10000)

# generate probability distribution
x_values = np.linspace(-0.99,0.99,10000)
y_values = f(x_values)

# sin(x) transformation
# y = ax +b transformation
plt.hist(np.sin(vector_uniform), bins=30, density=True, label="Transformation Data")
plt.plot(x_values, y_values, label="PDF")
plt.legend(loc="upper right")
plt.xlabel("x", fontname='Times New Roman')
plt.ylabel("PDF", fontname='Times New Roman')
plt.show()
plt.clf()


# part 2 (positive clipping)
def h(x):
    if np.sin(x) <= 0.7:
        return np.sin(x)
    else:
        return 0.7

def h_2(x):
    if abs(np.sin(x)) <= 0.7:
        return np.sin(x)
    elif np.sin(x) > 0.7:
        return 0.7
    else:
        return -0.7

def f_1(x):
    if x < 0.69:
        return f(x)
    else:
        pass

def f_2(x):
    if abs(x) < 0.69:
        return f(x)
    else:
        pass

fig, ax = plt.subplots(1,2, figsize=(13, 6))


(patches, bins, n) = ax[0].hist([h(i) for i in vector_uniform], bins=30, density=True, label="Transformation Data")
bin_area = bins[1]-bins[0]
ax[0].plot(x_values, [f_1(i) for i in x_values], label="PDF")
ax[0].plot(0.7,0.253/bin_area, marker=".", color="cornflowerblue")
ax[0].legend(loc="upper center")
ax[0].set_xlabel("x", fontname='Times New Roman')
ax[0].set_ylabel("PDF", fontname='Times New Roman')
ax[0].set_title("One-sided Clipping", fontname='Times New Roman', fontsize=13)


# part 2 (realistic symmetric clipping)
(patches, bins, n) = ax[1].hist([h_2(i) for i in vector_uniform], bins=30, density=True, label="Transformation Data")
bin_area = bins[1]-bins[0]
ax[1].plot(x_values, [f_2(i) for i in x_values], label="PDF")
ax[1].plot(0.7,0.253/bin_area, marker=".", color="cornflowerblue")
ax[1].plot(-0.7,0.253/bin_area,marker=".", color="cornflowerblue")
ax[1].legend(loc="upper center")
ax[1].set_xlabel("x", fontname='Times New Roman')
ax[1].set_ylabel("PDF", fontname='Times New Roman')
ax[1].set_title("Symmetric Clipping", fontname='Times New Roman', fontsize=13)

plt.show()