import numpy as np
import argparse
from astropy.io import fits
import XCFfit as fit
from XCFreader import GRun
from XCFGraph import GFigure, GPDMaps
from XCFgpd import GPD
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

def Gaussian(x,A,mu,sig):
    return A*np.exp(-(x-mu)**2/(2*sig**2))

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str, help = "Path to fits files", required=True)
args = parser.parse_args()

Data = GRun(args.path)
pha = Data.GetPha()
height, edges = np.histogram(pha, bins=400, range=[0, 20000])
centers = edges[:-1] + (edges[1]-edges[0])/2
mask = np.where((centers>5370) & (centers<7550))
par, _ = curve_fit(Gaussian, centers[mask], height[mask], p0=[height.max(), 5370+(7550-5370)/2, (7550-5370)/2] , sigma=np.sqrt(height[mask])) #self.GF, x, y, sigma=errY, p0=params, bounds=bounds

plt.figure()
plt.hist(edges[:-1], bins=edges, weights=height, label='Pha spectrum')
#plt.plot(centers, Gaussian(centers, par[0], par[1], par[2]))
plt.xlabel('Pha', fontsize=24, fontweight='bold')
plt.ylabel('Counts', fontsize=24, fontweight='bold')
plt.grid()
plt.legend(fontsize=24)
plt.show()