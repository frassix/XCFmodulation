import numpy as np
import argparse
from XCFreader import GRun
from XCFGraph import GFigure, GPDMaps
from XCFspurious import SpuriousMap
from XCFgpd import GPD

def MeanAndSigma(q0, q90, sigma0, sigma90):
        mean_sp = ( np.mean(q0)+np.mean(q90) )/2.
        mean_int = ( np.mean(q0)-np.mean(q90) )/2.
        sigma = ( np.sqrt( sigma0**2 + sigma90**2 ) )/2.
        return mean_sp, mean_int, sigma

parser = argparse.ArgumentParser()
parser.add_argument("--deg0", type=str, help = "Path to fits files with angle 0deg", required=True)
parser.add_argument("--deg90", type=str, help = "Path to fits files with angle 90deg", required=True)
parser.add_argument("--p", type=str, help = "if set to 1 draws the projection plots", default=False)
args = parser.parse_args()

print('\n' + '-'*100 + 'HOT SPOT' + '-'*100)
run_00 = GRun(args.deg0)
run_90 = GRun(args.deg90)

par_gs_00, cov_gs_00, _, chi_gs_00 = run_00.CutUnderPeak()
par_gs_90, cov_gs_90, _, chi_gs_90 = run_90.CutUnderPeak()

par_xy_00, _, gaussian, chi_xy_00 = run_00.FindGaussCenter()
par_xy_90, _,        _, chi_xy_90 = run_90.FindGaussCenter()

xc = (par_xy_00[0][1] + par_xy_90[0][1])/2
yc = (par_xy_00[1][1] + par_xy_90[1][1])/2
a  = (par_xy_00[0][2] + par_xy_90[1][2])/2.
b  = (par_xy_00[1][2] + par_xy_90[0][2])/2.


run_00.EllipticCut(xc, yc, a, b)
run_90.EllipticCut(xc, yc, a, b)

Map_00 = GPDMaps(run_00, figsize=(12,12))
Map_00(args.p, suptitle='Run 0 deg')
Map_00.DrawEllipse(xc, yc, a, b)
Map_00.ContourCounts()
Map_00.AxesLimits([xc-1.5*a, xc+1.5*a], [yc-1.5*b, yc+1.5*b])
par_gs_00[0] = par_gs_00[0]/10000000
Map_00.AddFunctionPlot([par_gs_00[1]-3*par_gs_00[2], par_gs_00[1]+3*par_gs_00[2]], gaussian, par_gs_00, 'Gaussian', np.sqrt(np.diag(cov_gs_00)), chi_gs_00, 1)

Map_90 = GPDMaps(run_90, figsize=(12,12))
Map_90(args.p, suptitle='Run 90 deg')
Map_90.DrawEllipse(xc, yc, a, b)
Map_90.ContourCounts()
Map_90.AxesLimits([xc-1.5*a, xc+1.5*a], [yc-1.5*b, yc+1.5*b])
par_gs_90[0] = par_gs_90[0]/10000000
Map_90.AddFunctionPlot([par_gs_90[1]-3*par_gs_90[2], par_gs_90[1]+3*par_gs_90[2]], gaussian, par_gs_90, 'Gaussian', np.sqrt(np.diag(cov_gs_90)), chi_gs_90, 1)
Map_00.GDraw()

q_00 = run_00.GetQ()
u_00 = run_00.GetU()
q_90 = run_90.GetQ()
u_90 = run_90.GetU()

sigmaQ_00, sigmaU_00 = run_00.CalculateQUError( np.mean(q_00), np.mean(u_00), len(u_00) )
sigmaQ_90, sigmaU_90 = run_90.CalculateQUError( np.mean(q_90), np.mean(u_90), len(u_90) )
Q, Qint, sQ = MeanAndSigma(q_00, q_90, sigmaQ_00, sigmaQ_90)
U, Uint, sU = MeanAndSigma(u_00, u_90, sigmaU_00, sigmaU_90)
MeanSpurious = np.sqrt( Q**2 +  U**2 )
MeanIntrinsec = np.sqrt( Qint**2 +  Uint**2 )
SigmaSpurious = ( np.sqrt( (Q*sQ)**2 + (U*sU)**2 ) )/MeanSpurious
SigmaIntrinsec = ( np.sqrt( (Qint*sQ)**2 + (Uint*sU)**2 ) )/MeanIntrinsec
print('-'*70+f'\nHOT SPOT MEAN SPURIOUS MODULATION VALUE: {MeanSpurious:.4f} ± {SigmaSpurious:.4f}')
print(f'HOT SPOT MEAN INTRINSEC MODULATION VALUE: {MeanIntrinsec:.4f} ± {SigmaIntrinsec:.4f}\n'+'-'*70+'\n')
print('-'*208 + '\n')

Sp = SpuriousMap('/data2/XCF/gpd_data/GPD35/Ott24/Corrected/0Deg/TMC5/', '/data2/XCF/gpd_data/GPD35/Ott24/Corrected/90Deg/TMC5/', suffix='spot')
Sp.HotSpot(xc, yc, a , b)
spurious, sigma, _, _ = Sp.MeanSpuriousValue()
Sp.ShowBoth()
Q_sp, U_sp = Sp.MakeSpuriousMap(True, False)
myGPD = GPD()
myGPD.ReadSpurious('/home/users/frassa/SpMap_spot.fits')
run_00.MakeMaps(myGPD)
Map_corrected_00 = GPDMaps(run_00, figsize=(12,12))
Map_corrected_00(args.p, suptitle='Run 0 deg')
Map_corrected_00.DrawEllipse(xc, yc, a, b)
Map_corrected_00.ContourCounts()
Map_corrected_00.AxesLimits([xc-1.5*a, xc+1.5*a], [yc-1.5*b, yc+1.5*b])
Map_corrected_00.GDraw()
