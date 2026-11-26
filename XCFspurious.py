"""Spurious (and intrinsic) modulation maps from two runs taken at 0 and 90 degrees; can be run as a script."""
import argparse
import os

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

from XCFGraph import GFigure, GPDMaps
from XCFreader import GRun


def StrToBool(value):
    """Converts command line strings like 1/0, true/false, yes/no to bool."""
    if isinstance(value, bool):
        return value
    if value.lower() in ('1', 'true', 't', 'yes', 'y', 'on'):
        return True
    if value.lower() in ('0', 'false', 'f', 'no', 'n', 'off'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean value, got {value!r}')


def MeanAndSigma(q0, q90, sigma0, sigma90):
    """Returns (spurious mean, intrinsic mean, sigma) of one Stokes parameter from the 0 and 90 deg runs."""
    mean_sp = (np.mean(q0)+np.mean(q90))/2.
    mean_int = (np.mean(q0)-np.mean(q90))/2.
    sigma = (np.sqrt(sigma0**2 + sigma90**2))/2.
    return mean_sp, mean_int, sigma


def CombineModulation(q_00, u_00, q_90, u_90, sigma_00, sigma_90):
    """Returns (MeanSp, SigmaSp, MeanInt, SigmaInt, Q, sQ, U, sU) from per-event q/u and (sigmaQ, sigmaU) of each run."""
    Q, Qint, sQ = MeanAndSigma(q_00, q_90, sigma_00[0], sigma_90[0])
    U, Uint, sU = MeanAndSigma(u_00, u_90, sigma_00[1], sigma_90[1])
    MeanSpurious = np.sqrt(Q**2 + U**2)
    MeanIntrinsec = np.sqrt(Qint**2 + Uint**2)
    SigmaSpurious = (np.sqrt((Q*sQ)**2 + (U*sU)**2))/MeanSpurious
    SigmaIntrinsec = (np.sqrt((Qint*sQ)**2 + (Uint*sU)**2))/MeanIntrinsec
    return MeanSpurious, SigmaSpurious, MeanIntrinsec, SigmaIntrinsec, Q, sQ, U, sU


class SpuriousMap():
    """Builds, plots and saves the spurious modulation map from a 0 deg and a 90 deg run."""

    def __init__(self, path0, path90, outdir='./', suffix=None):
        """Reads both runs, cuts under the PHA peak and sets the output file name (SpMap[_suffix].fits)."""
        self.Run_00 = GRun(path0)
        self.Run_90 = GRun(path90)
        fileName = 'SpMap.fits' if suffix is None else f'SpMap_{suffix}.fits'
        self.Path2File = os.path.join(outdir, fileName)
        self.meanMDP = None

        self.std_Cut00 = list(self.Run_00.CutUnderPeak())  # [par, cov, function, chi2]
        self.std_Cut90 = list(self.Run_90.CutUnderPeak())
        self.DoQU()

    def DoQU(self):
        """Copies the per-event q and u of both runs."""
        self.q_00 = self.Run_00.GetQ()
        self.u_00 = self.Run_00.GetU()
        self.q_90 = self.Run_90.GetQ()
        self.u_90 = self.Run_90.GetU()

    def DiagnosticGraph(self):
        """Plots the PHA distributions of both runs."""
        DCanvas = GFigure(1, 2)
        for position, (pha, angle) in enumerate(((self.Run_00.GetPha(), 0), (self.Run_90.GetPha(), 90))):
            DCanvas.AddHist(pha, 100, [pha.min(), pha.max()], label=rf'$Pha_{{{angle}}}$ distribution', position=position)
            DCanvas.AxesLabels(xlabel=r'Pha', ylabel='counts', position=position)
            DCanvas.TurnOnLegend(position=position)
            DCanvas.AxesLimits([0., 2*np.mean(pha)], [None, None], position)

    def HotSpot(self, xc, yc, a, b):
        """Keeps only the events inside the ellipse (xc, yc, a, b) in both runs."""
        self.Run_00.EllipticCut(xc, yc, a, b)
        self.Run_90.EllipticCut(xc, yc, a, b)
        self.DoQU()

    def MeanAndSigma(self, q0, q90, sigma0, sigma90):
        """Same as the module function MeanAndSigma (kept for compatibility)."""
        return MeanAndSigma(q0, q90, sigma0, sigma90)

    def MeanMdp(self):
        """Returns the mean MDP (times modulation factor) over the non-empty bins."""
        m = self._MdpMap()
        return np.mean(m[~np.isinf(m)])

    def MeanSpuriousValue(self):
        """Prints and returns (MeanSpurious, SigmaSpurious, MeanIntrinsic, SigmaIntrinsic) for |x|,|y|<7 mm."""
        phys_00 = self.Run_00.GetPhysicsMask()
        phys_90 = self.Run_90.GetPhysicsMask()
        q_00, u_00 = self.q_00[phys_00], self.u_00[phys_00]
        q_90, u_90 = self.q_90[phys_90], self.u_90[phys_90]
        sigma_00 = self.Run_00.CalculateQUError(np.mean(q_00), np.mean(u_00), len(phys_00))
        sigma_90 = self.Run_90.CalculateQUError(np.mean(q_90), np.mean(u_90), len(phys_90))
        MeanSpurious, SigmaSpurious, MeanIntrinsec, SigmaIntrinsec, Q, sQ, U, sU = \
            CombineModulation(q_00, u_00, q_90, u_90, sigma_00, sigma_90)
        print('-'*70+f'\nMEAN SPURIOUS MODULATION VALUE: {MeanSpurious:.4f} ± {SigmaSpurious:.4f}')
        print(f'MEAN Q: {Q:.4f} ± {sQ:.4f}')
        print(f'MEAN U: {U:.4f} ± {sU:.4f}')
        print(f'MEAN INTRINSEC MODULATION VALUE: {MeanIntrinsec:.4f} ± {SigmaIntrinsec:.4f}\n'+'-'*70+'\n')
        return MeanSpurious, SigmaSpurious, MeanIntrinsec, SigmaIntrinsec

    def MakeSpuriousMap(self, show=False, save=True):
        """Builds Q_sp=(Q0+Q90)/2 and U_sp, plots them (smoothed), optionally saves them; returns (Q_sp, U_sp)."""
        Q_00, U_00, Q_90, U_90, C_00, C_90 = self._RunMaps()
        Q_sp = (Q_00+Q_90)/2.
        U_sp = (U_00+U_90)/2.
        Q_err, U_err = self._MapErrors(Q_00, U_00, Q_90, U_90, C_00, C_90)
        firstCanvas = self._PlotMaps(Q_sp, U_sp, Q_err, U_err, C_00, C_90, 'sp', 'Spurious', smooth=True)
        if save:
            self.SaveMaps(Q_sp, U_sp, Q_err, U_err)
        if show:
            firstCanvas.GDraw()
        return Q_sp, U_sp

    def MakeIntrinsecMap(self, show=False, save=False):
        """Builds Q_in=(Q0-Q90)/2 and U_in, plots them (not smoothed); returns (Q_in, U_in). `save` is not used."""
        Q_00, U_00, Q_90, U_90, C_00, C_90 = self._RunMaps()
        Q_in = (Q_00-Q_90)/2.
        U_in = (U_00-U_90)/2.
        Q_err, U_err = self._MapErrors(Q_00, U_00, Q_90, U_90, C_00, C_90)
        firstCanvas = self._PlotMaps(Q_in, U_in, Q_err, U_err, C_00, C_90, 'in', 'Intrinsic', smooth=False)
        if show:
            firstCanvas.GDraw()
        return Q_in, U_in

    def MdpMap(self, nBins=100, show=False):
        """Plots the MDP map (4.29/sqrt(counts)) and its distribution, and stores the mean MDP."""
        print('\n----------------------MDP map added----------------------')
        MDP = GFigure(1, 2)
        MDP.SetFigureTitle(suptitle='MDP Study')
        m = self._MdpMap()
        self.meanMDP = np.mean(m[~np.isinf(m)])
        mapId = MDP.AddMap(m, vmax=m.max(), vmin=0., lim=7.5, position=0)
        MDP.AddColorBar(mapId, 'M', 0)
        MDP.AxesLabels('AbsX [mm]', 'AbsY [mm]', position=0)
        MDP.SetTitle('MDP Map', position=0)
        MDP.AddHist(np.where(np.isinf(m), -1, m).ravel(), nBins, [0., 1.], label='Modulation', position=1)
        MDP.AxesLabels(r'$\mu \cdot MDP$', 'Entries', position=1)
        MDP.SetTitle('M distribution', position=1)
        MDP.TurnOnGrid(1)
        if show:
            MDP.GDraw()
        print(f'Mean MDP value {self.meanMDP}')
        print('----------------------Done with MDP----------------------\n')

    def SaveMaps(self, Q_sp, U_sp, Q_err, U_err, numGPD=35):
        """Writes the maps to self.Path2File (HDUs: PRIMARY=[mu-sigma, mu+sigma] of the PHA peak, Q_SP, U_SP, M_SP, COUNTS, QERR, UERR)."""
        par_00 = self.std_Cut00[0]
        useful = [par_00[1]-par_00[2], par_00[1]+par_00[2]]
        hdr = fits.Header()
        hdr['GPD_NUM'] = numGPD
        hdr['PHA_BIN'] = "PHA_MIN, PHA_MAX"
        hdr['Comment'] = 'Q and U spurious map'

        hdul = fits.HDUList([
            fits.PrimaryHDU(data=useful, header=hdr),
            fits.ImageHDU(data=Q_sp, name='Q_SP'),
            fits.ImageHDU(data=U_sp, name='U_SP'),
            fits.ImageHDU(data=np.sqrt(Q_sp**2 + U_sp**2), name='M_SP'),
            fits.ImageHDU(data=(self.Run_00.GetCounts()+self.Run_90.GetCounts()), name='COUNTS'),
            fits.ImageHDU(data=Q_err, name='QERR'),
            fits.ImageHDU(data=U_err, name='UERR'),
        ])
        hdul.writeto(self.Path2File, overwrite=True)
        print(f'\nFILE SAVED: {self.Path2File}')
        self.PrintFileInfo()

    def PrintFileInfo(self):
        """Prints the HDU list of the saved file."""
        with fits.open(self.Path2File) as hdul:
            hdul.info()

    def ShowBoth(self):
        """Draws the standard maps of both runs with the PHA peak fit."""
        for Run, (par, cov, function, chi2) in ((self.Run_00, self.std_Cut00), (self.Run_90, self.std_Cut90)):
            Canvas = GPDMaps(Run)
            Canvas(False)
            Canvas.AddFunctionPlot([par[1]-4*par[2], par[1]+4*par[2]], function, par, sigma=np.sqrt(np.diag(cov)),
                                   label='Peak Cut', chi2=chi2, position=1)
        Canvas.GDraw()

    def TargetMdp(self, mdp):
        """Rebins all maps so that each bin reaches the requested MDP."""
        N = (4.29/mdp)**2
        BinContent = np.mean(self.Run_00.GetCounts()+self.Run_90.GetCounts())
        M = np.shape(self.Run_00.GetCounts())[0]
        target = int(N/BinContent)
        print(f"\n\nYou should aggregate {target} bins. So {M/target}x{M/target} bins.")
        print("------------------------------REBINNING ALL MAPS------------------------------\n")
        self.Run_00.RebinAll(int(M/target))
        self.Run_90.RebinAll(int(M/target))
        self.meanMDP = None

    # ---------------------------------------------------------------- internals
    def _MapErrors(self, Q_00, U_00, Q_90, U_90, C_00, C_90):
        """Returns the Q and U error maps, 0 where a run has <= 1 count."""
        ok = (C_00 > 1) & (C_90 > 1)
        with np.errstate(divide='ignore', invalid='ignore'):
            Q_err = np.where(ok, 0.5*np.sqrt((2-Q_00**2)/(C_00-1)+(2-Q_90**2)/(C_90-1)), 0)
            U_err = np.where(ok, 0.5*np.sqrt((2-U_00**2)/(C_00-1)+(2-U_90**2)/(C_90-1)), 0)
        return Q_err, U_err

    def _MdpMap(self):
        """Returns the MDP map 4.29/sqrt(counts_0 + counts_90) (inf in empty bins)."""
        with np.errstate(divide='ignore'):
            return 4.29/np.sqrt(self.Run_00.GetCounts()+self.Run_90.GetCounts())

    def _PlotMaps(self, Q, U, Q_err, U_err, C_00, C_90, tag, name, smooth):
        """Draws Q/U maps, error maps, Q/U/M/psi distributions; returns the first figure."""
        print(f"Making {name.lower()} map.\nBinned:{Q.shape[0]}x{Q.shape[1]}")
        if self.meanMDP is None:
            self.meanMDP = self.MeanMdp()
        filt = (lambda z, s: gaussian_filter(z, sigma=s)) if smooth else (lambda z, s: z)

        MapCanvas = GFigure(1, 2)
        for position, (Z, S) in enumerate(((Q, 'Q'), (U, 'U'))):
            mapId = MapCanvas.AddMap(filt(Z, 1), vmin=-0.3, vmax=0.3, lim=7.5, position=position)
            MapCanvas.AddColorBar(mapId, S, position=position)
            MapCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=position)
            MapCanvas.SetTitle(rf'${S}_{{{tag}}}$', position=position)

        ErrorCanvas = GFigure(1, 3)
        mapC = ErrorCanvas.AddMap(C_00+C_90, vmin=0, vmax=C_00.max()*2, lim=7.5, position=0)
        ErrorCanvas.AddColorBar(mapC, 'Counts', position=0)
        ErrorCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
        ErrorCanvas.SetTitle(r'$Counts_{0}\,+\,Counts_{90}$', position=0)
        for position, (E, S) in enumerate(((Q_err, 'Q'), (U_err, 'U')), start=1):
            mapId = ErrorCanvas.AddMap(E, vmin=-0.3, vmax=0.3, lim=7.5, position=position)
            ErrorCanvas.AddColorBar(mapId, rf'$\sigma_{S}$', position=position)
            ErrorCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=position)
            ErrorCanvas.SetTitle(rf'${S}_{{{name}}}\, error map$', position=position)

        for Z, E, S in ((Q, Q_err, 'Q'), (U, U_err, 'U')):
            Canvas = GFigure(1, 2)
            Canvas.AddHist(Z, 100, [-2., 2.], label=rf'${S}_{{{tag}}}$ distribution', position=0)
            Canvas.AxesLabels(xlabel=rf'${S}_{{{tag}}}$', ylabel='counts', position=0)
            Canvas.SetTitle(rf'${S}_{{{tag}}}$', position=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                Canvas.AddHist(np.abs(E/Z), 100, [0., 10], label=rf'rel ${S}err_{{{tag}}}$', position=1)
            Canvas.AxesLabels(xlabel=rf'$|\sigma_{S}/{S}|$', ylabel='counts', position=1)
            Canvas.SetTitle(rf'Relative ${S}err_{{{tag}}}$', position=1)

        M = np.sqrt(Q**2 + U**2)
        psi = 0.5*np.rad2deg(np.arctan2(U, Q))

        ModulationCanvas = GFigure(1, 2)
        mapM = ModulationCanvas.AddMap(filt(M, 1), vmin=0., vmax=1, lim=7.5, position=0)
        ModulationCanvas.AddColorBar(mapM, 'M', position=0)
        ModulationCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
        ModulationCanvas.SetTitle(rf'$M_{{{tag}}}$', position=0)
        ModulationCanvas.AddHist(M.ravel(), 150, [0., 2.], label=f'{name} modulation', position=1)
        ModulationCanvas.AddHist(ModulationCanvas.EdgeCut(M, 7), 150, [0., 2.], label=f'{name} modulation |x|,|y|<7 mm', position=1)
        ModulationCanvas.AxesLabels(xlabel=rf'$M_{{{tag}}}$', ylabel='Frequency', position=1)
        ModulationCanvas.SetTitle(rf'$M_{{{tag}}}$ distribution', position=1)
        ModulationCanvas.TurnOnLegend(position=1)
        ModulationCanvas.DrawVline(self.meanMDP, 0, 4000, 'mean MDP', position=1)
        ModulationCanvas.AxesLimits([0, 2.], [None, None], 1)

        AngleCanvas = GFigure(1, 2)
        mapPsi = AngleCanvas.AddMap(filt(psi, 2), vmin=-90., vmax=90., lim=7.5, position=0)
        AngleCanvas.AddColorBar(mapPsi, r'$\psi [deg]$', position=0)
        AngleCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
        AngleCanvas.SetTitle(rf'$\psi_{{{tag}}}$', position=0)
        AngleCanvas.AddHist(psi.ravel(), 180, [-90, 90], label=f'{name} angle', position=1)
        AngleCanvas.AddHist(AngleCanvas.EdgeCut(psi, 7), 180, [-90, 90], label=f'{name} angle |x|,|y|<7 mm', position=1)
        AngleCanvas.AxesLabels(xlabel=rf'$\psi_{{{tag}}}$', ylabel='Frequency', position=1)
        AngleCanvas.SetTitle(rf'$\psi_{{{tag}}}$ distribution', position=1)
        AngleCanvas.TurnOnLegend(position=1)

        print("Map done")
        return MapCanvas

    def _RunMaps(self):
        """Returns Q, U of the 0 deg run, Q, U of the 90 deg run and both counts maps."""
        return (self.Run_00.GetQMap(), self.Run_00.GetUMap(), self.Run_90.GetQMap(), self.Run_90.GetUMap(),
                self.Run_00.GetCounts(), self.Run_90.GetCounts())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Spurious modulation map from 0 and 90 deg runs.')
    parser.add_argument("--deg0", type=str, help="Path to fits file/dir with angle 0deg", required=True)
    parser.add_argument("--deg90", type=str, help="Path to fits file/dir with angle 90deg", required=True)
    parser.add_argument("--out", "--output-folder", type=str, help="Path to output dir (must exist)", default='./')
    parser.add_argument("--s", "--name-suffix", type=str, help="File name suffix (result SpMap_suffix.fits)", default=None)
    parser.add_argument("--d", "--diagnostic", type=StrToBool, nargs='?', const=True, default=False,
                        help="Shows helpful PHA plots (--d or --d 1)")
    parser.add_argument("--intrinsic", action='store_true', help="Also make the intrinsic modulation map")
    parser.add_argument("--target-mdp", type=float, default=None, help="Rebin the maps to reach this MDP per bin")
    parser.add_argument("--no-show", action='store_true', help="Do not open the plot windows (only compute and save)")
    args = parser.parse_args()

    Sp = SpuriousMap(args.deg0, args.deg90, args.out, args.s)
    if args.target_mdp is not None:
        Sp.TargetMdp(args.target_mdp)
    Sp.MdpMap()
    Sp.MeanSpuriousValue()
    if args.d:
        Sp.DiagnosticGraph()
    if args.intrinsic:
        Sp.MakeIntrinsecMap()
    Sp.MakeSpuriousMap(show=not args.no_show)
