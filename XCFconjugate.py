"""Conjugate modulation: kernel elements K_{m,0}, K_{m,+2}, K_{m,-2} from runs taken at different polarization angles.

Model (note "Extracting the kernel elements from two arbitrary polarization angles", eq. 3):
    M_m(theta) = K_m0 + A/2 K_m2 exp(+2i theta) + A/2 K_m-2 exp(-2i theta),   M_m = <exp(i m phi)>
theta is the polarization angle in the detector frame. Two angles need K_m0 from outside (spurious map or a value);
three or more angles determine all three elements, with a chi2 test when the system is over-determined.
"""
import argparse
from dataclasses import dataclass, field

import numpy as np
from astropy.io import fits

from XCFGraph import GFigure
from XCFgpd import GPD
from XCFreader import GRun

V_ISO = 0.5  # per-event variance of cos(m phi) and sin(m phi) in the isotropic approximation (eq. 49)
TERMS = {'known0': ('K2', 'Km2'), 'full': ('K0', 'K2', 'Km2'), 'noconj': ('K0', 'K2')}
MODE_TEXT = {'known0': 'K_m0 known (subtracted)', 'full': 'all three elements fitted', 'noconj': 'K_m-2 assumed 0 (section 9)'}


# ---------------------------------------------------------------- math (usable without data)
def Model(theta, K0, K2, Km2, A=1.):
    """Eq. (3): moment M_m expected at detector polarization angle theta [rad]."""
    return K0 + 0.5*A*K2*np.exp(2j*theta) + 0.5*A*Km2*np.exp(-2j*theta)


def TwoAngleClosedForm(d1, d2, theta1, theta2, A=1.):
    """Eqs. (33)-(37): (K_m2, K_m-2) from the reduced moments d_j = M_j - K_m0 of two runs (closed form, for checks)."""
    c1, s1, c2, s2 = np.cos(2*theta1), np.sin(2*theta1), np.cos(2*theta2), np.sin(2*theta2)
    L = c1*s2 - c2*s1
    P = (s2*np.real(d1) - s1*np.real(d2))/L
    S = (c1*np.real(d2) - c2*np.real(d1))/L
    T = (s2*np.imag(d1) - s1*np.imag(d2))/L
    R = (c1*np.imag(d2) - c2*np.imag(d1))/L
    return ((P+R) + 1j*(T-S))/A, ((P-R) + 1j*(T+S))/A


def DesignMatrix(thetas, A, terms):
    """Real design matrix: rows (Re d_1, Im d_1, Re d_2, ...), columns (Re, Im) of each unknown in `terms`."""
    rows = []
    for theta in thetas:
        z = np.exp(2j*theta)
        coefficient = {'K0': 1.+0j, 'K2': 0.5*A*z, 'Km2': 0.5*A*np.conj(z)}
        re, im = [], []
        for term in terms:
            g = coefficient[term]
            re += [g.real, -g.imag]
            im += [g.imag, g.real]
        rows += [re, im]
    return np.array(rows)


def SolveKernel(thetas, A, d, cov, terms):
    """Weighted least squares of d_j = model (any leading batch dims); returns (params, cov, chi2, dof, estimator L)."""
    X = DesignMatrix(thetas, A, terms)
    nRuns = len(thetas)
    y = np.stack([np.real(d), np.imag(d)], axis=-1).reshape(*np.shape(d)[:-1], 2*nRuns)
    Cinv = np.linalg.inv(cov)                                        # (..., nRuns, 2, 2)
    W = np.zeros(Cinv.shape[:-3] + (2*nRuns, 2*nRuns))
    for j in range(nRuns):
        W[..., 2*j:2*j+2, 2*j:2*j+2] = Cinv[..., j, :, :]
    F = np.einsum('ia,...ij,jb->...ab', X, W, X)
    if np.any(np.linalg.cond(F) > 1e12):
        raise np.linalg.LinAlgError('singular system')
    pcov = np.linalg.inv(F)
    L = np.einsum('...ab,ib,...ij->...aj', pcov, X, W)
    p = np.einsum('...aj,...j->...a', L, y)
    res = y - np.einsum('ia,...a->...i', X, p)
    chi2 = np.einsum('...i,...ij,...j->...', res, W, res)
    return p, pcov, chi2, 2*nRuns - X.shape[1], L


def Propagate(func, x, cov, h=1e-7):
    """Value and error of func(x) with linear propagation of cov (numerical gradient)."""
    x = np.asarray(x, dtype=float)
    grad = np.array([(func(x + h*e) - func(x - h*e))/(2*h) for e in np.eye(len(x))])
    return func(x), np.sqrt(max(grad @ cov @ grad, 0.))


def Label(name, m):
    """Math-text plot label of a derived quantity of harmonic m."""
    return {'|K_m,+2|': rf'$|K_{{{m},+2}}|$', '|K_m,-2|': rf'$|K_{{{m},-2}}|$', '|K_m,0|': rf'$|K_{{{m},0}}|$',
            'r': r'$r = |K_{m,-2}|/|K_{m,+2}|$'.replace('m', str(m)), 'Delta psi [deg]': r'$\Delta\psi$ [deg]'}.get(name, name)


def WrapDeg(angle):
    """Wraps an angle in degrees to (-180, 180]."""
    return -((-angle + 180.) % 360. - 180.)


@dataclass
class KernelResult:
    """Kernel elements of one harmonic, with statistical and systematic (from K_m0) covariances."""
    m: int
    mode: str
    thetas: np.ndarray
    A: float
    K0: complex                      # K_m0 subtracted (known0) or fitted (full, noconj); spurious map not included
    K2: complex
    Km2: complex
    params: np.ndarray               # real parameters, order given by TERMS[mode]
    cov: np.ndarray                  # statistical covariance of params
    syst: np.ndarray                 # systematic covariance of params (error on K_m0 / spurious map)
    chi2: float
    dof: int
    d: np.ndarray                    # reduced moments of each run (spurious map and known K_m0 subtracted)
    dcov: np.ndarray                 # their 2x2 covariances
    N: np.ndarray                    # events of each run
    derived: dict = field(default_factory=dict)  # name -> (value, stat, syst)


# ---------------------------------------------------------------- analysis class
class ConjugateModulation():
    """Measures K_m,+2 (modulation factor) and the conjugate K_m,-2 from runs at known polarization angles."""

    def __init__(self, runs, A=1., harmonics=(2,), spmap=None, k0=None, mode='auto',
                 clocking=False, phiLab=0., peakCut=True, nSigma=3., edgeCut=False):
        """runs: list of (angle [deg], GRun or path); spmap: spurious map file (m=2); k0: {m: (K_m0, sigma)}."""
        self.A = A
        self.harmonics = tuple(harmonics)
        self.mode = mode
        self.k0 = {m: (complex(v[0]), complex(v[1]) if len(v) > 1 else 0j) for m, v in (k0 or {}).items()}
        angles = np.array([float(a) for a, _ in runs])
        self.thetas = np.radians(phiLab - angles if clocking else angles)   # eq. (4): theta = phi_lab - chi
        self.runs = []
        for _, run in runs:
            if not isinstance(run, GRun):
                run = GRun(run)
                if peakCut:
                    run.CutUnderPeak(nSigma)
                if edgeCut:
                    run.CutTheEdges()
            self.runs.append(run)
        if len(self.runs) < 2:
            raise ValueError('At least two runs are needed')
        self.spGPD = self._LoadSpurious(spmap) if spmap else None
        self.spPath = spmap
        self.results = {}
        self.CheckAngles()

    # ------------------------------------------------------------ inputs
    def _LoadSpurious(self, path):
        """Reads the spurious map (Q_SP, U_SP and errors, in units of 2cos/2sin) on its own grid."""
        nBins = fits.getheader(path, 1)['NAXIS1']
        sp = GPD(mapbins=nBins)
        sp.ReadSpurious(path)
        return sp

    def CheckAngles(self):
        """Prints separation, noise amplification G = 1/|sin 2Delta| and warnings (eqs. 22-25)."""
        print('-'*30 + 'CONJUGATE MODULATION' + '-'*30)
        for theta, run in zip(np.degrees(self.thetas), self.runs):
            print(f'  theta = {theta:8.3f} deg   N = {len(run.GetPhi())}   {run.GetPath()}')
        if len(self.thetas) == 2:
            delta = np.degrees(self.thetas[0] - self.thetas[1])
            s2d = abs(np.sin(2*np.radians(delta)))
            print(f'  Delta = {delta:.3f} deg, |sin 2Delta| = {s2d:.3f}, G = {1/s2d if s2d > 0 else np.inf:.2f}')
            if s2d < 0.2:
                print('  WARNING: Delta close to k*90 deg: K_m,+2 and K_m,-2 cannot be separated (eq. 22)')
            if abs(WrapDeg(3*delta)/3) < 1.:
                print('  WARNING: Delta is a multiple of 60 deg: lattice systematics cannot be separated (eq. 25)')
        else:
            X = DesignMatrix(self.thetas, self.A, TERMS['full'])
            print(f'  {len(self.thetas)} angles, condition number of the full system: {np.linalg.cond(X):.2f}')

    def Moments(self, m, binGrid=None):
        """Returns (d, cov, N) of each run: mean of exp(i m phi) minus the spurious map (m=2), and its covariance."""
        d, cov, N = [], [], []
        for run in self.runs:
            w = np.exp(1j*m*run.GetPhi())
            if m == 2 and self.spGPD is not None:
                idx = run.BinIndexes(self.spGPD)
                w = w - 0.5*(self.spGPD.GetQ().ravel()[idx] + 1j*self.spGPD.GetU().ravel()[idx])
            if binGrid is None:
                d.append(w.mean())
                cov.append(np.cov(np.vstack((w.real, w.imag)))/len(w))
                N.append(len(w))
            else:
                bd, bc, bn = self._BinnedMoments(w, run.BinIndexes(binGrid), binGrid)
                d.append(bd)
                cov.append(bc)
                N.append(bn)
        axis = 0 if binGrid is None else -1
        return np.stack(d, axis=axis), np.stack(cov, axis=-3), np.stack(N, axis=axis)

    def _BinnedMoments(self, w, idx, grid):
        """Per-bin mean of w and covariance of the mean (NaN where a bin has < 3 events)."""
        nb = grid.GetRows()*grid.GetColumns()
        n = np.bincount(idx, minlength=nb).astype(float)
        x, y = w.real, w.imag
        sx, sy = np.bincount(idx, x, nb), np.bincount(idx, y, nb)
        sxx, syy, sxy = np.bincount(idx, x*x, nb), np.bincount(idx, y*y, nb), np.bincount(idx, x*y, nb)
        with np.errstate(divide='ignore', invalid='ignore'):
            mx, my = sx/n, sy/n
            norm = np.where(n > 2, 1/((n-1)*n), np.nan)
            cxx, cyy, cxy = (sxx - n*mx*mx)*norm, (syy - n*my*my)*norm, (sxy - n*mx*my)*norm
        cov = np.stack([np.stack([cxx, cxy], -1), np.stack([cxy, cyy], -1)], -2)
        return mx + 1j*my, cov, n

    def _SpuriousSystematic(self, m):
        """Covariance (2N x 2N) of the reduced moments due to the error on K_m0 (map errors and/or given sigma)."""
        nRuns = len(self.runs)
        S = np.zeros((2*nRuns, 2*nRuns))
        if m == 2 and self.spGPD is not None:
            nb = self.spGPD.GetRows()*self.spGPD.GetColumns()
            w = np.array([np.bincount(run.BinIndexes(self.spGPD), minlength=nb)/len(run.GetPhi()) for run in self.runs])
            for k, err in ((0, self.spGPD.GetQerr()), (1, self.spGPD.GetUerr())):
                sigma2 = (0.5*np.asarray(err, float).ravel())**2
                S[k::2, k::2] += (w*sigma2) @ w.T   # Cov(eps_j, eps_k) = sum_b w_jb w_kb sigma_b^2
        if m in self.k0:
            sig = self.k0[m][1]
            S[0::2, 0::2] += sig.real**2
            S[1::2, 1::2] += sig.imag**2 if sig.imag else sig.real**2
        return S

    def EmptyMapFraction(self):
        """Fraction of events of each run falling in spurious-map bins without information (error = 0)."""
        empty = (self.spGPD.GetQerr().ravel() == 0)
        return [np.mean(empty[run.BinIndexes(self.spGPD)]) for run in self.runs]

    # ------------------------------------------------------------ solving
    def ChooseMode(self, m):
        """auto: 'full' with >= 3 runs, 'known0' with 2 runs if K_m0 is available (map for m=2 or given)."""
        if self.mode != 'auto':
            return self.mode
        if len(self.runs) >= 3:
            return 'full'
        if (m == 2 and self.spGPD is not None) or m in self.k0:
            return 'known0'
        raise ValueError(f'm={m}: two runs need K_{m},0 from outside: give a spurious map (m=2), k0 for m={m}, '
                         f'a third angle, or mode="noconj" (assumes K_{m},-2 = 0)')

    def Solve(self, m):
        """Solves harmonic m, stores and returns a KernelResult."""
        mode = self.ChooseMode(m)
        terms = TERMS[mode]
        d, dcov, N = self.Moments(m)
        k0 = self.k0.get(m, (0j, 0j))[0] if mode == 'known0' else 0j
        d = d - k0
        try:
            p, pcov, chi2, dof, L = SolveKernel(self.thetas, self.A, d, dcov, terms)
        except np.linalg.LinAlgError:
            raise ValueError(f'm={m}, mode {mode}: the system is singular for these angles '
                             '(two angles 90 deg apart cannot separate K_m,+2 and K_m,-2; eqs. 9 and 22)')
        syst = L @ self._SpuriousSystematic(m) @ L.T
        values = dict(zip(terms, p[0::2] + 1j*p[1::2]))
        res = KernelResult(m, mode, self.thetas, self.A, values.get('K0', k0), values['K2'], values.get('Km2', 0j),
                           p, pcov, syst, float(chi2), dof, d, dcov, N)
        self._Derived(res, terms)
        self.results[m] = res
        return res

    def _Derived(self, res, terms):
        """Adds mu=|K_m2|, |K_m-2|, r and Delta psi (with stat and syst errors) to the result."""
        i2 = 2*terms.index('K2')
        functions = {'|K_m,+2|': lambda x: np.hypot(x[i2], x[i2+1])}
        if 'Km2' in terms:
            im = 2*terms.index('Km2')
            functions['|K_m,-2|'] = lambda x: np.hypot(x[im], x[im+1])
            functions['r'] = lambda x: np.hypot(x[im], x[im+1])/np.hypot(x[i2], x[i2+1])
            dpsi0 = np.degrees(np.arctan2(res.params[i2+1], res.params[i2]) - np.arctan2(res.params[im+1], res.params[im]))
            ref = WrapDeg(dpsi0)  # evaluate the angle close to its value to avoid the +-180 jump in the gradient
            functions['Delta psi [deg]'] = lambda x: ref + WrapDeg(np.degrees(np.arctan2(x[i2+1], x[i2]) - np.arctan2(x[im+1], x[im])) - ref)
        if 'K0' in terms:
            functions['|K_m,0|'] = lambda x: np.hypot(x[0], x[1])
        for name, f in functions.items():
            value, stat = Propagate(f, res.params, res.cov)
            _, syst = Propagate(f, res.params, res.syst)
            res.derived[name] = (value, stat, syst)

    def SolveAll(self):
        """Solves every requested harmonic (skipping the ones that cannot be solved) and prints the summaries."""
        for m in self.harmonics:
            try:
                self.Solve(m)
            except ValueError as e:
                print(f'\nSKIPPED m={m}: {e}')
                continue
            print(self.Summary(m))
        return self.results

    # ------------------------------------------------------------ output
    def Summary(self, m):
        """Returns a text summary of harmonic m."""
        r = self.results[m]
        terms = TERMS[r.mode]
        stat = np.sqrt(np.diag(r.cov))
        syst = np.sqrt(np.diag(r.syst))
        lines = ['\n' + '='*90, f'HARMONIC m = {m}   mode: {MODE_TEXT[r.mode]}   runs: {len(r.thetas)}   A = {self.A}']
        if m == 2 and self.spGPD is not None:
            empty = ', '.join(f'{100*f:.2f}%' for f in self.EmptyMapFraction())
            lines.append(f'Spurious map subtracted event by event: {self.spPath}   (events in empty map bins: {empty})')
        if r.mode == 'known0':
            lines.append(f'K_{m},0 subtracted (besides the map): {r.K0.real:+.6f} {r.K0.imag:+.6f}i')
        lines.append(f'{"":18s}{"value":>12s}{"stat":>11s}{"syst":>11s}')
        for k, term in enumerate(terms):
            name = {'K0': f'K_{m},0', 'K2': f'K_{m},+2', 'Km2': f'K_{m},-2'}[term]
            for part, j in (('Re', 2*k), ('Im', 2*k+1)):
                lines.append(f'{part + " " + name:18s}{r.params[j]:+12.6f}{stat[j]:11.6f}{syst[j]:11.6f}')
        for name, (value, s1, s2) in r.derived.items():
            lines.append(f'{name.replace("m", str(m), 1) if name.startswith("|K") else name:18s}{value:+12.6f}{s1:11.6f}{s2:11.6f}')
        if 'Delta psi [deg]' in r.derived:
            lines.append('(Delta psi depends on the absolute angle calibration; moduli and r do not: remark 1.2)')
        if r.dof > 0:
            lines.append(f'chi2 / dof = {r.chi2:.2f} / {r.dof}')
        if len(r.thetas) == 2 and r.mode == 'known0':
            eps = np.sqrt(np.trace(self._SpuriousSystematic(m)[:2, :2]))
            delta = r.thetas[0] - r.thetas[1]
            if eps > 0:
                ncross = 4*V_ISO*np.cos(delta)**2/(eps**2*np.sin(2*delta)**2)
                lines.append(f'|eps(K_{m},0)| ~ {eps:.2e}: statistics stop helping above N_cross ~ {ncross:.2e} events (eq. 61)')
        lines.append('='*90)
        return '\n'.join(lines)

    def RequiredEvents(self, r, nSigma=3., m=2, mu=None):
        """Eq. (55): total events needed to see a conjugate ratio r at nSigma (two angles; 3+ equispaced: sin=1)."""
        if mu is None:
            mu = self.results[m].derived['|K_m,+2|'][0] if m in self.results else 0.3
        s2d = abs(np.sin(2*(self.thetas[0]-self.thetas[1]))) if len(self.thetas) == 2 else 1.
        return 4*nSigma**2*V_ISO/(r**2*mu**2*self.A**2*s2d**2)

    def PlotSummary(self, nBins=180):
        """Phi distributions with their 2+4 harmonic description, and the reduced moments with the fitted ellipse."""
        Fig = GFigure(1, 1 + len(self.results))
        for run, theta in zip(self.runs, np.degrees(self.thetas)):
            phi = run.GetPhi()
            Fig.AddHist(phi, nBins, (-np.pi, np.pi), alpha=0.4, label=f'{theta:.1f} deg', normalized=True)
            x = np.linspace(-np.pi, np.pi, 400)
            c2, c4 = np.mean(np.exp(2j*phi)), np.mean(np.exp(4j*phi))
            Fig.AddPlot(x, (1 + 2*np.real(c2*np.exp(-2j*x)) + 2*np.real(c4*np.exp(-4j*x)))/(2*np.pi), '-', label='m=2,4 moments')
        Fig.AxesLabels(r'$\phi$ [rad]', 'Density')
        Fig.SetTitle('Phi distributions')
        Fig.TurnOnLegend()
        for position, (m, r) in enumerate(self.results.items(), start=1):
            t = np.linspace(0, np.pi, 300)
            model = r.K0*(r.mode != 'known0') + 0.5*self.A*(r.K2*np.exp(2j*t) + r.Km2*np.exp(-2j*t))
            Fig.AddPlot(model.real, model.imag, '-', label=r'model $\theta\in[0,180)$', position=position)
            for j, theta in enumerate(np.degrees(r.thetas)):
                Fig.AddErrorbar(r.d[j].real, r.d[j].imag, np.sqrt(r.dcov[j, 0, 0]), np.sqrt(r.dcov[j, 1, 1]),
                                fmt='o', label=fr'run $\theta$={theta:.1f}', position=position)
            ratio = r.derived.get('r')
            Fig.SetTitle(f'm={m}' + (f'   r = {ratio[0]:.4f} ± {ratio[1]:.4f}' if ratio else ''), position=position)
            Fig.AxesLabels(f'Re $d_{m}$', f'Im $d_{m}$', position=position)
            Fig.SetEqual(position)
            Fig.TurnOnGrid(position)
            Fig.TurnOnLegend(position)
        return Fig

    def MakeMaps(self, m, nBins=30, minCounts=200):
        """Solves harmonic m in every bin of an nBins x nBins grid; returns dict of maps (NaN where not solvable)."""
        mode = self.ChooseMode(m)
        terms = TERMS[mode]
        grid = GPD(mapbins=nBins)
        d, dcov, N = self.Moments(m, binGrid=grid)                      # (nb, nRuns), (nb, nRuns, 2, 2), (nb, nRuns)
        if mode == 'known0':
            d = d - self.k0.get(m, (0j, 0j))[0]
        ok = np.all(N >= minCounts, axis=-1) & np.all(np.isfinite(dcov), axis=(-3, -2, -1))
        maps = {name: np.full(nBins*nBins, np.nan) for name in ('|K_m,+2|', '|K_m,-2|', 'r', 'Delta psi [deg]', '|K_m,0|')}
        maps['counts'] = N.sum(axis=-1)
        if ok.any():
            p, _, _, _, _ = SolveKernel(self.thetas, self.A, d[ok], dcov[ok], terms)
            K = {t: p[:, 2*k] + 1j*p[:, 2*k+1] for k, t in enumerate(terms)}
            maps['|K_m,+2|'][ok] = np.abs(K['K2'])
            if 'Km2' in K:
                maps['|K_m,-2|'][ok] = np.abs(K['Km2'])
                maps['r'][ok] = np.abs(K['Km2'])/np.abs(K['K2'])
                maps['Delta psi [deg]'][ok] = WrapDeg(np.degrees(np.angle(K['K2']) - np.angle(K['Km2'])))
            if 'K0' in K:
                maps['|K_m,0|'][ok] = np.abs(K['K0'])
        print(f'm={m}: solved {ok.sum()} of {ok.size} bins ({nBins}x{nBins}, at least {minCounts} events per run)')
        return {name: z.reshape(nBins, nBins) for name, z in maps.items()}

    def PlotMaps(self, m, nBins=30, minCounts=200):
        """Draws counts, |K_m,+2|, r and Delta psi (or |K_m,0|) maps of harmonic m."""
        maps = self.MakeMaps(m, nBins, minCounts)
        Fig = GFigure(2, 2)
        Fig.SetFigureTitle(f'Kernel maps, m={m}  ({nBins}x{nBins} bins)')
        last = 'Delta psi [deg]' if np.isfinite(maps['Delta psi [deg]']).any() else '|K_m,0|'
        for position, name in enumerate(('counts', '|K_m,+2|', 'r', last)):
            z = maps[name]
            finite = z[np.isfinite(z)]
            vmin, vmax = (-180., 180.) if name == last and last.startswith('Delta') else \
                (0., np.percentile(finite, 99) if finite.size else 1.)
            mapId = Fig.AddMap(z, vmin=vmin, vmax=vmax, position=position)
            label = Label(name, m)
            Fig.AddColorBar(mapId, label, position=position)
            Fig.AxesLabels('X [mm]', 'Y [mm]', position=position)
            Fig.SetTitle(label, position=position)
        return Fig


# ---------------------------------------------------------------- self test
def SelfTest():
    """Reproduces the worked example of section 6 (A=1, Delta=45 deg) with the closed form and the general solver."""
    K0, K2, Km2 = 0.0040*np.exp(1j*np.radians(40)), 0.1800+0j, 0.0180*np.exp(1j*np.radians(110))
    thetas = np.radians([0., 45.])
    M = Model(thetas, K0, K2, Km2)
    print('Step 1, measured moments (note: Q1=+0.089986 U1=+0.011028 Q2=+0.011521 U2=+0.095649)')
    for t, mj in zip((0, 45), M):
        print(f'  theta={t:2d}  Q={mj.real:+.6f}  U={mj.imag:+.6f}')
    d = M - K0
    a, b = TwoAngleClosedForm(d[0], d[1], *thetas)
    p, _, _, _, _ = SolveKernel(thetas, 1., d, np.tile(np.eye(2), (2, 1, 1)), TERMS['known0'])
    print(f'Closed form : K2 = {a.real:+.6f}{a.imag:+.6f}i   K-2 = {b.real:+.6f}{b.imag:+.6f}i')
    print(f'Solver      : K2 = {p[0]:+.6f}{p[1]:+.6f}i   K-2 = {p[2]:+.6f}{p[3]:+.6f}i')
    print('Note (eq.44): K2 = +0.180000+0.000000i   K-2 = -0.006156+0.016914i')
    r, dpsi = abs(b)/abs(a), WrapDeg(np.degrees(np.angle(a) - np.angle(b)))
    print(f'r = {r:.6f} (0.1),  Delta psi = {dpsi:.4f} deg (-110)')
    full = Model(np.radians([0., 90., 45.]), K0, K2, Km2)
    p3, _, _, _, _ = SolveKernel(np.radians([0., 90., 45.]), 1., full, np.tile(np.eye(2), (3, 1, 1)), TERMS['full'])
    print(f'0/90/45 programme (section 9.2): K0 = {p3[0]:+.6f}{p3[1]:+.6f}i  K2 = {p3[2]:+.6f}{p3[3]:+.6f}i  '
          f'K-2 = {p3[4]:+.6f}{p3[5]:+.6f}i')
    worst = max(abs(a-K2), abs(b-Km2), abs(p[0]+1j*p[1]-K2), abs(p[2]+1j*p[3]-Km2),
                abs(p3[0]+1j*p3[1]-K0), abs(p3[2]+1j*p3[3]-K2), abs(p3[4]+1j*p3[5]-Km2))
    print(f'Worst error on the injected values: {worst:.1e}  ->  {"OK" if worst < 1e-12 else "FAILED"}')
    return worst < 1e-12


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Conjugate modulation (K_m,+2, K_m,-2) from runs at different polarization angles.')
    parser.add_argument('--run', nargs=2, action='append', metavar=('ANGLE', 'PATH'), default=[],
                        help='Polarization angle [deg] in the detector frame (or clocking angle with --clocking) and file/dir; repeat for every run')
    parser.add_argument('--A', type=float, default=1., help='Polarization degree of the beam (default 1)')
    parser.add_argument('--harmonics', type=int, nargs='+', default=[2], help='Harmonics m to analyse (default 2)')
    parser.add_argument('--spmap', type=str, default=None, help='Spurious map file (SpMap*.fits): gives K_2,0 event by event')
    parser.add_argument('--k0', nargs='+', action='append', default=[], metavar='M RE IM [SIGMA]',
                        help='Known K_m,0 for harmonic M (and its error SIGMA, systematic); repeatable')
    parser.add_argument('--mode', choices=('auto', 'known0', 'full', 'noconj'), default='auto',
                        help='auto: full with >=3 runs, known0 with 2 runs; noconj assumes K_m,-2 = 0')
    parser.add_argument('--clocking', action='store_true', help='ANGLE values are clocking angles chi: theta = phi_lab - chi')
    parser.add_argument('--phi-lab', type=float, default=0., help='Beam polarization angle in the lab [deg] (with --clocking)')
    parser.add_argument('--no-peak-cut', action='store_true', help='Do not keep only the events under the PHA peak')
    parser.add_argument('--nsigma', type=float, default=3., help='PHA peak cut width in sigma (default 3)')
    parser.add_argument('--edge-cut', action='store_true', help='Keep only |x|,|y| < 7 mm')
    parser.add_argument('--maps', type=int, default=0, metavar='NBINS', help='Also solve in NBINS x NBINS bins and draw the maps')
    parser.add_argument('--min-counts', type=int, default=200, help='Minimum events per run in a map bin (default 200)')
    parser.add_argument('--target-r', type=float, default=None, help='Print the events needed to detect this r at 3 sigma (eq. 55)')
    parser.add_argument('--no-show', action='store_true', help='Do not open the plot windows')
    parser.add_argument('--selftest', action='store_true', help='Run the worked example of the note and exit')
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(0 if SelfTest() else 1)
    if len(args.run) < 2:
        parser.error('give at least two --run ANGLE PATH')
    k0 = {}
    for item in args.k0:
        if len(item) not in (3, 4):
            parser.error('--k0 needs M RE IM [SIGMA]')
        k0[int(item[0])] = (complex(float(item[1]), float(item[2])), float(item[3]) if len(item) == 4 else 0.)

    CM = ConjugateModulation([(float(a), p) for a, p in args.run], A=args.A, harmonics=args.harmonics, spmap=args.spmap,
                             k0=k0, mode=args.mode, clocking=args.clocking, phiLab=args.phi_lab,
                             peakCut=not args.no_peak_cut, nSigma=args.nsigma, edgeCut=args.edge_cut)
    results = CM.SolveAll()
    if args.target_r:
        for m in results:
            print(f'm={m}: events needed for r={args.target_r} at 3 sigma: {CM.RequiredEvents(args.target_r, 3., m):.3e}')
    if results:
        Fig = CM.PlotSummary()
        if args.maps:
            for m in results:
                CM.PlotMaps(m, args.maps, args.min_counts)
        if not args.no_show:
            Fig.GDraw()
