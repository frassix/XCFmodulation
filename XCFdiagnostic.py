"""Plots the PHA spectrum of a run, optionally with a gaussian fit of the peak."""
import argparse

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from XCFreader import GRun


def Gaussian(x, A, mu, sig):
    """Gaussian A*exp(-(x-mu)^2/(2 sig^2))."""
    return A*np.exp(-(x-mu)**2/(2*sig**2))


parser = argparse.ArgumentParser(description='PHA spectrum of a run.')
parser.add_argument("--path", type=str, help="Path to fits file or directory", required=True)
parser.add_argument("--fit", action='store_true', help="Fit and draw a gaussian on the peak")
parser.add_argument("--fit-min", type=float, default=5370, help="Lower PHA of the fit window (default 5370)")
parser.add_argument("--fit-max", type=float, default=7550, help="Upper PHA of the fit window (default 7550)")
args = parser.parse_args()

Data = GRun(args.path)
height, edges = np.histogram(Data.GetPha(), bins=400, range=[0, 20000])
centers = edges[:-1] + (edges[1]-edges[0])/2

plt.figure()
plt.hist(edges[:-1], bins=edges, weights=height, label='Pha spectrum')
if args.fit:
    window = (centers > args.fit_min) & (centers < args.fit_max)
    p0 = [height.max(), (args.fit_min+args.fit_max)/2, (args.fit_max-args.fit_min)/2]
    par, cov = curve_fit(Gaussian, centers[window], height[window], p0=p0, sigma=np.sqrt(height[window]))
    print(f'Peak fit: A = {par[0]:.1f}, mu = {par[1]:.1f} ± {np.sqrt(cov[1, 1]):.1f}, sigma = {par[2]:.1f} ± {np.sqrt(cov[2, 2]):.1f}')
    plt.plot(centers, Gaussian(centers, *par), label='Gaussian fit')
plt.xlabel('Pha', fontsize=24, fontweight='bold')
plt.ylabel('Counts', fontsize=24, fontweight='bold')
plt.grid()
plt.legend(fontsize=24)
plt.show()
