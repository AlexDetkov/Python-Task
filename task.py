import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

N = 10000	# number of samples from Gaussian distribution

np.random.seed(1)	# set constant random seed to get same results
points = np.random.normal(size=N)

n, bins, _ = plt.hist(points, bins="auto", label="Data")

# calculate gaussian fit
def gaus(x,a,x0,sigma):
	return a*np.exp(-(x-x0)**2/(2*sigma**2))

bins_center = 0.5*(bins[1:] + bins[:-1])	
popt,pcov = curve_fit(gaus,bins_center,n)
plt.plot(bins_center, gaus(bins_center,*popt), label='Gaussian Fit')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Gaussian Samples with Gaussian Fit')
plt.legend()
plt.savefig('histogram.png')
plt.savefig('histogram.pdf')
plt.show()
