"""Hot spot study: finds the beam spot of the 0/90 deg runs, cuts an ellipse around it and measures the modulation inside."""
import argparse

import numpy as np

from XCFGraph import GPDMaps
from XCFgpd import GPD
from XCFreader import GRun
from XCFspurious import CombineModulation, SpuriousMap, StrToBool


def SpotMaps(run, xc, yc, a, b, projections, suptitle):
    """Draws the standard maps of run zoomed on the ellipse (xc, yc, a, b) with the count contours."""
    Maps = GPDMaps(run, figsize=(12, 12))
    Maps(projections, suptitle=suptitle)
    Maps.DrawEllipse(xc, yc, a, b)
    Maps.ContourCounts()
    Maps.AxesLimits([xc-1.5*a, xc+1.5*a], [yc-1.5*b, yc+1.5*b])
    return Maps


parser = argparse.ArgumentParser(description='Spurious/intrinsic modulation of the beam hot spot.')
parser.add_argument("--deg0", type=str, help="Path to fits files with angle 0deg", required=True)
parser.add_argument("--deg90", type=str, help="Path to fits files with angle 90deg", required=True)
parser.add_argument("--p", type=StrToBool, nargs='?', const=True, default=False,
                    help="Draws the projection plots (--p or --p 1)")
parser.add_argument("--sp-deg0", type=str, default='/data2/XCF/gpd_data/GPD35/Ott24/Corrected/0Deg/TMC5/',
                    help="0 deg data used for the spot spurious map")
parser.add_argument("--sp-deg90", type=str, default='/data2/XCF/gpd_data/GPD35/Ott24/Corrected/90Deg/TMC5/',
                    help="90 deg data used for the spot spurious map")
parser.add_argument("--spmap", type=str, default='/home/users/frassa/SpMap_spot.fits',
                    help="Spurious map file used to correct the 0 deg run")
args = parser.parse_args()

print('\n' + '-'*100 + 'HOT SPOT' + '-'*100)
run_00 = GRun(args.deg0)
run_90 = GRun(args.deg90)

par_gs_00, cov_gs_00, _, chi_gs_00 = run_00.CutUnderPeak()
par_gs_90, cov_gs_90, _, chi_gs_90 = run_90.CutUnderPeak()

par_xy_00, _, gaussian, chi_xy_00 = run_00.FindGaussCenter()
par_xy_90, _,        _, chi_xy_90 = run_90.FindGaussCenter()

# Ellipse: mean centre of the two runs; semi-axes mix X of one run with Y of the other (as in the original analysis)
xc = (par_xy_00[0][1] + par_xy_90[0][1])/2
yc = (par_xy_00[1][1] + par_xy_90[1][1])/2
a = (par_xy_00[0][2] + par_xy_90[1][2])/2.
b = (par_xy_00[1][2] + par_xy_90[0][2])/2.

run_00.EllipticCut(xc, yc, a, b)
run_90.EllipticCut(xc, yc, a, b)

for run, par, cov, chi2, title in ((run_00, par_gs_00, cov_gs_00, chi_gs_00, 'Run 0 deg'),
                                   (run_90, par_gs_90, cov_gs_90, chi_gs_90, 'Run 90 deg')):
    Maps = SpotMaps(run, xc, yc, a, b, args.p, title)
    par[0] = par[0]/10000000  # rough rescale of the amplitude to the normalized PHA histogram
    Maps.AddFunctionPlot([par[1]-3*par[2], par[1]+3*par[2]], gaussian, par, 'Gaussian', np.sqrt(np.diag(cov)), chi2, 1)
Maps.GDraw()

q_00, u_00 = run_00.GetQ(), run_00.GetU()
q_90, u_90 = run_90.GetQ(), run_90.GetU()
sigma_00 = run_00.CalculateQUError(np.mean(q_00), np.mean(u_00), len(u_00))
sigma_90 = run_90.CalculateQUError(np.mean(q_90), np.mean(u_90), len(u_90))
MeanSpurious, SigmaSpurious, MeanIntrinsec, SigmaIntrinsec, *_ = CombineModulation(q_00, u_00, q_90, u_90, sigma_00, sigma_90)
print('-'*70+f'\nHOT SPOT MEAN SPURIOUS MODULATION VALUE: {MeanSpurious:.4f} ± {SigmaSpurious:.4f}')
print(f'HOT SPOT MEAN INTRINSEC MODULATION VALUE: {MeanIntrinsec:.4f} ± {SigmaIntrinsec:.4f}\n'+'-'*70+'\n')
print('-'*208 + '\n')

Sp = SpuriousMap(args.sp_deg0, args.sp_deg90, suffix='spot')
Sp.HotSpot(xc, yc, a, b)
Sp.MeanSpuriousValue()
Sp.ShowBoth()
Q_sp, U_sp = Sp.MakeSpuriousMap(True, False)

myGPD = GPD()
myGPD.ReadSpurious(args.spmap)
run_00.MakeMaps(myGPD)
SpotMaps(run_00, xc, yc, a, b, args.p, 'Run 0 deg').GDraw()
