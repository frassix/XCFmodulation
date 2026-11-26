"""Reads GPD event files (single file or a whole directory) and builds q/u, modulation and Q/U maps."""
import argparse
import glob
import os

import astropy.io.fits as pf
import numpy as np

import XCFfit as fit
import XCFgpd as gpd

GAUSSIAN = "[0]*exp(-(x-[1])**2/(2*[2]**2))"


def PrintFitsHeaders(path):
    """Prints every header (primary and extensions) of one FITS file."""
    print(f'\n===== {path}')
    with pf.open(path) as hdul:
        for i, hdu in enumerate(hdul):
            print(f"\nHeader {i+1} (primary or extension):")
            for keyword, value in hdu.header.items():
                print(f"{keyword}: {value}")
    print()


class GRun():
    """One acquisition: events read from FITS files, Stokes q/u per event and binned Q/U maps."""

    def __init__(self, file_path, sCut=True):
        """Reads file_path (file or directory), applies the standard cuts if sCut and builds the maps."""
        self.path = file_path
        self.Cut = sCut
        self.totalLiveTime = 0.
        self.modulation = None
        self.Phi = np.array([])
        self.Absx = np.array([])
        self.Absy = np.array([])
        self.Pha = np.array([])
        self.mapGPD = gpd.GPD()   # binning of the maps
        self.spurious = None      # GPD whose spurious map is subtracted (None = no correction)
        self.FindFiles()
        self.CalculateQUC()
        self.MakeMaps()

    def ApplyMask(self, mask, redo=True):
        """Keeps only the selected events and, if redo, recomputes q/u and the maps."""
        self.Phi = self.Phi[mask]
        self.Absx = self.Absx[mask]
        self.Absy = self.Absy[mask]
        self.Pha = self.Pha[mask]
        print(f'Selected data: {len(self.Pha)}')
        if redo:
            self.CalculateQUC()
            self.MakeMaps()

    def CalculateQUError(self, q, u, N):
        """Returns the statistical errors (sigmaQ, sigmaU) of the mean q and u over N events."""
        sigmaQ = np.sqrt((2-q**2)/(N-1))
        sigmaU = np.sqrt((2-u**2)/(N-1))
        return sigmaQ, sigmaU

    def CalculateQUC(self):
        """Computes the raw per-event q, u and the total (uncorrected) modulation and polarization angle Psi."""
        self.q_raw = 2*np.cos(2*self.Phi)
        self.u_raw = 2*np.sin(2*self.Phi)
        self.q, self.u = self.q_raw, self.u_raw  # MakeMaps applies the spurious correction, if any
        mean_q = np.mean(self.q_raw)
        mean_u = np.mean(self.u_raw)
        self.modulation = np.sqrt(mean_q**2 + mean_u**2)
        self.Psi = 0.5*np.arctan2(mean_u, mean_q)

    def CutTheEdges(self, limX=7., limY=7.):
        """Removes the events outside the rectangle |x|<limX, |y|<limY."""
        mask = (np.abs(self.Absx) < limX) & (np.abs(self.Absy) < limY)
        print(f'Cutting all events out of X:[{-limX}, {limX}], Y:[{-limY}, {limY}]')
        self.ApplyMask(mask, True)

    def CutUnderPeak(self, nSigma=3.):
        """Fits the PHA peak with a gaussian and keeps events within nSigma; returns (par, cov, function, chi2)."""
        print('-'*30 + 'CUTTING PHA UNDER PEAK' + '-'*30)
        Gaussian = fit.GFit(GAUSSIAN)
        x, y, par0, bounds = self.MakeGaussian(self.Pha, 200, [0, 30000])
        param, cov = Gaussian(x, y, errY=np.sqrt(y), params=par0, bounds=bounds)
        self.ApplyMask(np.abs(self.Pha-param[1]) <= nSigma*param[2], True)
        return param, cov, Gaussian.GetFunction(), Gaussian.GetChi2()

    def EllipticCut(self, xc, yc, a, b, inside=True):
        """Keeps the events inside (or outside, if inside=False) the ellipse centred in (xc, yc) with axes a, b."""
        if a == 0 or b == 0:
            a = 1
            b = 1
            print("ERROR: invalid value for elliptic shape. Setting a=b=1")
        r2 = ((self.Absx-xc)/a)**2 + ((self.Absy-yc)/b)**2
        if inside:
            mask = r2 <= 1
            print(f"Cutting data inside the ellipse: [(x-{xc:.2})/{a:.2}]^2 + [(y-{yc:.2})/{b:.2}]^2 <= 1")
        else:
            mask = r2 > 1
            print(f"Cutting data outside the ellipse: [(x-{xc:.2})/{a:.2}]^2 + [(y-{yc:.2})/{b:.2}]^2 > 1")
        self.ApplyMask(mask, True)

    def FindFiles(self):
        """Reads self.path: one FITS file or every *.fits file in a directory."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Given path '{self.path}' does not exist")
        if os.path.isfile(self.path):
            print("\n" + "-"*100 + "\n" + "-"*100 + f"\nReading single file: {self.path}")
            self._PrintCutInfo()
            self.ReadOneFile(file=self.path)
        else:
            print("\n" + "-"*100 + "\n" + "-"*100 + f"\nReading all files in dir: {self.path}")
            self._PrintCutInfo()
            self.StackAllFiles()
        print('Done reading\n-------------------------------------')
        self.PrintInfo()

    def FindGaussCenter(self):
        """Fits the X and Y projections with gaussians; returns ([par_x, par_y], [cov_x, cov_y], function, [chi2_x, chi2_y])."""
        print('-'*30 + 'FINDING GAUSSIAN CENTER' + '-'*30)
        Gaussian = fit.GFit(GAUSSIAN)
        print('X center position')
        bX, hX, par0_x, bounds_x = self.MakeGaussian(self.Absx, 300, [-7.5, 7.5], 1)
        param_x, cov_x = Gaussian(bX, hX, errY=np.sqrt(hX), params=par0_x, bounds=bounds_x)
        chi2_x = Gaussian.GetChi2()
        print('Y center position')
        bY, hY, par0_y, bounds_y = self.MakeGaussian(self.Absy, 300, [-7.5, 7.5], 1)
        param_y, cov_y = Gaussian(bY, hY, errY=np.sqrt(hY), params=par0_y, bounds=bounds_y)
        return [param_x, param_y], [cov_x, cov_y], Gaussian.GetFunction(), [chi2_x, Gaussian.GetChi2()]

    def FunctionCut(self, expression, param, above=False, ax='x'):
        """Keeps events above/below the curve y=f(x) (ax='x') or x=f(y) (ax='y'), f given as string."""
        func = fit.GF1(expression=expression)
        if ax not in ('x', 'X', 'y', 'Y'):
            print('Invalid ax in FunctionCut: setting ax to x')
            ax = 'x'
        if ax in ('x', 'X'):
            free, dependent = self.Absx, self.Absy
        else:
            free, dependent = self.Absy, self.Absx
        curve = func(free, *param)
        mask = dependent >= curve if above else dependent <= curve
        print(f'Cutting data following {expression}')
        self.ApplyMask(mask, True)

    def GetAbsX(self):
        """Returns the event X positions [mm]."""
        return self.Absx

    def GetAbsY(self):
        """Returns the event Y positions [mm]."""
        return self.Absy

    def GetCounts(self):
        """Returns the counts map."""
        return self.counts

    def GetIndexes(self):
        """Returns the map bin index of each event."""
        return self.bin_indices

    def GetModulation(self):
        """Returns the total modulation."""
        return self.modulation

    def GetPhysicsMask(self):
        """Returns the indexes of the events with |x|<7 and |y|<7."""
        return np.where((np.abs(self.Absx) < 7) & (np.abs(self.Absy) < 7))[0]

    def GetPath(self):
        """Returns the input path."""
        return self.path

    def GetPha(self):
        """Returns the event PHA."""
        return self.Pha

    def GetPhi(self):
        """Returns the event emission angle DETPHI2 [rad]."""
        return self.Phi

    def GetPsi(self):
        """Returns the total polarization angle [rad]."""
        return self.Psi

    def GetQ(self):
        """Returns the per-event q (spurious corrected, if a correction is set)."""
        return self.q

    def GetQMap(self):
        """Returns the Q map."""
        return self.Q

    def GetU(self):
        """Returns the per-event u (spurious corrected, if a correction is set)."""
        return self.u

    def GetUMap(self):
        """Returns the U map."""
        return self.U

    def LinearCut(self, min, max, ax='x'):
        """Keeps events with min <= X <= max (ax='x') or min <= Y <= max (any other ax)."""
        if ax in ('x', 'X'):
            mask = (self.Absx >= min) & (self.Absx <= max)
            print(f'Cutting data: {min}<=AbsX<={max}')
        else:
            mask = (self.Absy >= min) & (self.Absy <= max)
            print(f'Cutting data: {min}<=AbsY<={max}')
        self.ApplyMask(mask, True)

    def MakeGaussian(self, x, nBins, range, n=2):
        """Histograms x and returns (centers, heights above half max, initial params, bounds) for a gaussian fit."""
        h, b = np.histogram(x, bins=nBins, range=range)
        centers = b[:-1] + (b[1] - b[0]) / 2.
        Amp = h.max()
        mu = np.average(centers, weights=h)
        sig = np.sqrt(np.average((centers-mu)**2, weights=h))
        params = [Amp, mu, sig]
        bounds = ([Amp*0.9, mu-n*sig, 0.], [Amp*1.1, mu+n*sig, 2*n*sig])
        print(params)
        RegionOfFit = h >= h.max()/2.
        return centers[RegionOfFit], h[RegionOfFit], params, bounds

    def MakeMaps(self, myGPD=None):
        """Builds the Q, U, counts maps; a given myGPD becomes the spurious correction (applied once, on the raw q/u)."""
        print("Making Q and U maps")
        if myGPD is not None:
            self.spurious = myGPD
            self.mapGPD = myGPD
        # always start again from the raw values: the correction can never be subtracted twice
        self.q, self.u = self.q_raw, self.u_raw
        if self.spurious is not None:
            print("Subtracting spurious modulation map")
            sp_indices = self._BinIndexes(self.spurious)
            self.q = self.q_raw - self.spurious.GetQ().ravel()[sp_indices]
            self.u = self.u_raw - self.spurious.GetU().ravel()[sp_indices]

        x_, y_, _, _ = self.mapGPD.MapLimits()
        nBins = x_*y_
        self.bin_indices = self._BinIndexes(self.mapGPD)
        # bincount is much faster than np.add.at and sums in the same order
        counts = np.bincount(self.bin_indices, minlength=nBins).astype(float)
        q_tot = np.bincount(self.bin_indices, weights=self.q, minlength=nBins)
        u_tot = np.bincount(self.bin_indices, weights=self.u, minlength=nBins)
        filled = counts > 0
        self.Q = np.divide(q_tot, counts, out=np.zeros(nBins), where=filled).reshape(x_, y_)
        self.U = np.divide(u_tot, counts, out=np.zeros(nBins), where=filled).reshape(x_, y_)
        self.counts = counts.reshape(x_, y_)
        print("Maps Ready")

    def PrintHead(self):
        """Prints every header of the input file(s)."""
        for path in self._FitsFiles() if os.path.isdir(self.path) else [self.path]:
            PrintFitsHeaders(path)

    def PrintInfo(self):
        """Prints number of events, live time and maximum PHA."""
        print(
            f'\nData from {self.path}'
            f'\nTotal number of events: {len(self.Pha)}'
            f'\nTotal live-time: {self.totalLiveTime:.5} h'
            f'\nPha Max: {self.Pha.max()}'
        )

    def ReadOneFile(self, file):
        """Reads one FITS file and appends its events to the current ones."""
        self._Append([self._ReadEvents(file)])

    def RebinAll(self, Nbins):
        """Rebuilds the maps with Nbins x Nbins bins (a spurious correction already set is kept)."""
        self.mapGPD = gpd.GPD(mapbins=Nbins)
        self.MakeMaps()

    def ResetSpurious(self):
        """Removes the spurious correction and rebuilds the maps from the raw q/u."""
        self.spurious = None
        self.MakeMaps()

    def StackAllFiles(self):
        """Reads every *.fits file in the directory and stacks the events."""
        files = self._FitsFiles()
        if not files:
            raise FileNotFoundError(f"No .fits files found in '{self.path}'")
        print(f'Reading files: {files}')
        self._Append([self._ReadEvents(f) for f in files])

    def StdCut(self, df):
        """Standard selection (|x|,|y|<7.5, PHA>0, NUM_CLU>0, LIVETIME>15, TRK_SIZE>0); also adds the file live time."""
        data = df.data
        self.totalLiveTime += (data['TIME'].max()-data['TIME'].min())/3600
        return ((np.abs(data['ABSX']) < 7.5) & (np.abs(data['ABSY']) < 7.5) & (data['PHA'] > 0.)
                & (data['NUM_CLU'] > 0.) & (data['LIVETIME'] > 15.) & (data['TRK_SIZE'] > 0.))

    def VoidMask(self, df):
        """Selection that keeps every event."""
        pha = df.data['PHA']
        return pha >= pha.min()

    def _Append(self, chunks):
        """Concatenates the (phi, x, y, pha) chunks to the stored events in a single step."""
        self.Phi = np.concatenate([self.Phi] + [c[0] for c in chunks])
        self.Absx = np.concatenate([self.Absx] + [c[1] for c in chunks])
        self.Absy = np.concatenate([self.Absy] + [c[2] for c in chunks])
        self.Pha = np.concatenate([self.Pha] + [c[3] for c in chunks])

    def _BinIndexes(self, grid):
        """Returns the flat map bin index of each event on the grid of the GPD `grid`."""
        x_, y_, x_range, y_range = grid.MapLimits()
        x_indices = np.digitize(self.Absx, x_range)-1
        y_indices = np.digitize(self.Absy, y_range)-1
        outside = (x_indices < 0) | (x_indices >= x_) | (y_indices < 0) | (y_indices >= y_)
        if outside.any():
            raise ValueError(f'{np.count_nonzero(outside)} events are outside the map (|x| or |y| >= {grid.GetLimits()} mm): '
                             'use the standard cuts or CutTheEdges() before making the maps')
        return x_indices * y_ + y_indices

    def _FitsFiles(self):
        """Returns the sorted list of *.fits files in the input directory."""
        return sorted(glob.glob(os.path.join(self.path, '*.fits')))

    def _PrintCutInfo(self):
        """Prints which selection is used."""
        if self.Cut:
            print('StdCuts applyed: PHA>0 & NUM_CLU>0 & LIVETIME>15 & TRK_SIZE>0')
        else:
            print('No cuts applyed')

    def _ReadEvents(self, file):
        """Returns the selected (phi, x, y, pha) arrays of one FITS file."""
        with pf.open(file) as data_f:
            df = data_f['EVENTS']
            mask = self.StdCut(df) if self.Cut else self.VoidMask(df)
            return (df.data['DETPHI2'][mask], df.data['ABSX'][mask],
                    df.data['ABSY'][mask], df.data['PHA'][mask])


if __name__ == '__main__':
    from XCFGraph import GPDMaps

    parser = argparse.ArgumentParser(description='Quick look of a run: counts map, PHA and phi distributions.')
    parser.add_argument('path', help='FITS file or directory with FITS files')
    parser.add_argument('--no-cut', action='store_true', help='Do not apply the standard cuts')
    parser.add_argument('--peak', action='store_true', help='Also keep only the events under the PHA peak (3 sigma)')
    args = parser.parse_args()

    Run = GRun(args.path, sCut=not args.no_cut)
    if args.peak:
        Run.CutUnderPeak()
    Maps = GPDMaps(Run)
    Maps(True, suptitle=args.path)
    Maps.GDraw()
