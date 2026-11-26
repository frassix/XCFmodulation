# XCFmodulation

Analysis tools for GPD (Gas Pixel Detector) calibration runs taken at the X-ray Calibration Facility (XCF).

They read the event FITS files and compute the Stokes parameters, modulation and polarization angle of each event, plus binned Q/U maps. With a run at **0°** and one at **90°** they build the **spurious modulation map**, the map of fake modulation the detector adds by itself.

## Requirements

Python 3.8+ and:

```bash
pip install -r requirements.txt
```

(`numpy`, `scipy`, `astropy`, `matplotlib`)

## Input data

Each event file is a FITS file with an `EVENTS` extension that has the columns
`DETPHI2`, `ABSX`, `ABSY`, `PHA`, `NUM_CLU`, `LIVETIME`, `TRK_SIZE`, `TIME`.

Wherever a path is asked for, you can give **one file** or **a directory**. With a directory, every `*.fits` file inside is read and stacked.

Standard cuts, applied on reading (`GRun(path, sCut=False)` turns them off):
`|ABSX| < 7.5 mm`, `|ABSY| < 7.5 mm`, `PHA > 0`, `NUM_CLU > 0`, `LIVETIME > 15`, `TRK_SIZE > 0`.

## Files

| File | What it does |
|---|---|
| `XCFreader.py` | `GRun`: reads a run, applies cuts (PHA peak, edges, ellipse, lines, functions), computes q/u and the Q/U/counts maps |
| `XCFspurious.py` | `SpuriousMap`: spurious and intrinsic maps from 0°/90° runs, MDP map, saves `SpMap*.fits` (**main script**) |
| `XCFgpd.py` | `GPD`: detector geometry and spurious maps read from file; `Track`: draws single event tracks |
| `XCFGraph.py` | `GFigure`: small matplotlib wrapper; `GPDMaps`: standard plots of a run |
| `XCFfit.py` | `GF1`: functions written as strings (`"[0] + [1]*x"`); `GFit`: iterative chi2 fit |
| `HotSpot.py` | Modulation inside the beam spot (elliptic cut around the gaussian centre) |
| `Diagnostic.py` | Q/U maps and q/u distributions before and after the 7 mm edge cut |
| `XCFdiagnostic.py` | PHA spectrum, with an optional gaussian fit of the peak |
| `MapComparison.py` | Compares two spurious maps (ratio, Z compatibility, distributions) |
| `ShowEqMaps.py` | `EqMap`: ASIC equalization maps (`EQ_SLOPE` per pixel): map, distribution, column/row projections, ratio of two maps |
| `SpMap.fits` | Example spurious map (GPD 35, 300x300 bins) |

## Command line usage

Every script prints its options with `-h`.

**Spurious map** (writes `<out>/SpMap_<suffix>.fits` and opens the plots):

```bash
python XCFspurious.py --deg0 /data/GPD35/0Deg/ --deg90 /data/GPD35/90Deg/ --out ./ --s MoEn
```

Options: `--d` PHA diagnostic plots · `--intrinsic` also makes the intrinsic map · `--target-mdp 0.08` rebins to reach this MDP per bin · `--no-show` computes and saves without opening any window (useful on a server).

**Quick look of one run:**

```bash
python XCFreader.py /data/GPD35/0Deg/run.fits --peak
```

**Other scripts:**

```bash
python Diagnostic.py --inFile /data/GPD35/0Deg/run.fits
```

```bash
python XCFdiagnostic.py --path /data/GPD35/0Deg/ --fit --fit-min 5370 --fit-max 7550
```

```bash
python MapComparison.py --map1 SpMap_low.fits --map2 SpMap_high.fits
```

```bash
python HotSpot.py --deg0 /data/0Deg/ --deg90 /data/90Deg/ --p --sp-deg0 /data/TMC5/0Deg/ --sp-deg90 /data/TMC5/90Deg/ --spmap SpMap_spot.fits
```

```bash
python ShowEqMaps.py flat_A.fits --head --projections
```

```bash
python ShowEqMaps.py flat_A.fits flat_B.fits --ratio flat_A.fits flat_B.fits
```

```bash
python XCFgpd.py /data/run_with_pixels.fits --event 49
```

## Python example

```python
import numpy as np
from XCFreader import GRun
from XCFgpd import GPD
from XCFGraph import GPDMaps
from XCFspurious import SpuriousMap

# 1) one run: read, keep only the events under the PHA peak, look at it
run = GRun('/data/GPD35/0Deg/')                  # file or directory
par, cov, gauss, chi2 = run.CutUnderPeak(nSigma=3)
print('modulation:', run.GetModulation(), ' angle [deg]:', np.degrees(run.GetPsi()))

maps = GPDMaps(run)
maps(projections=True, suptitle='0 deg run')     # counts map, PHA, phi + X/Y projections
spot, spot_cov, gauss_xy, chi2_xy = run.FindGaussCenter()
proj = maps.GetProjection()
mu, sig = spot[0][1], spot[0][2]
proj.AddFunctionPlot([mu-3*sig, mu+3*sig], gauss_xy, spot[0], label='X fit',
                     sigma=np.sqrt(np.diag(spot_cov[0])), chi2=chi2_xy[0], position=0)
maps.Save('run0.png')                            # or maps.GDraw() to show every figure
proj.Save('run0_projections.png')

# 2) other cuts (each one recomputes q/u and the maps)
run.CutTheEdges(7., 7.)                          # |x|,|y| < 7 mm
run.EllipticCut(xc=0.2, yc=-0.1, a=2., b=1.5)    # keep inside the ellipse
run.FunctionCut("[0] + [1]*x", (0., 1.), above=True)   # keep y >= x

# 3) spurious map from 0/90 deg runs
sp = SpuriousMap('/data/GPD35/0Deg/', '/data/GPD35/90Deg/', outdir='./', suffix='MoEn')
sp.MdpMap()
sp.MeanSpuriousValue()                           # prints mean spurious/intrinsic modulation
Q_sp, U_sp = sp.MakeSpuriousMap(show=False, save=True)   # -> ./SpMap_MoEn.fits

# 4) correct another run with that map
gpd = GPD()
gpd.ReadSpurious('./SpMap_MoEn.fits')
run2 = GRun('/data/GPD35/45Deg/')
run2.MakeMaps(gpd)                               # q, u, Q, U are now spurious-subtracted (only once)
```

## What is computed

For each event with emission angle φ (`DETPHI2`):

* `q = 2 cos 2φ`, `u = 2 sin 2φ`
* modulation `m = sqrt(<q>² + <u>²)`, angle `ψ = ½ atan2(<u>, <q>)`
* error of the mean: `σ_q = sqrt((2 − <q>²)/(N − 1))` (same for u)
* maps: events are binned in 300×300 bins over ±7.5 mm, and `Q`, `U` are the mean q, u in each bin

From the 0° and 90° runs:

* spurious: `Q_sp = (Q_0 + Q_90)/2`, `U_sp = (U_0 + U_90)/2`
* intrinsic: `Q_in = (Q_0 − Q_90)/2`, `U_in = (U_0 − U_90)/2`
* per-bin error: `½ sqrt((2 − Q_0²)/(N_0 − 1) + (2 − Q_90²)/(N_90 − 1))`
* MDP map: `4.29 / sqrt(N_0 + N_90)`

### Spurious map file (`SpMap*.fits`)

| HDU | Name | Content |
|---|---|---|
| 0 | PRIMARY | `[μ − σ, μ + σ]` of the PHA peak of the 0° run |
| 1 | Q_SP | spurious Q map |
| 2 | U_SP | spurious U map |
| 3 | M_SP | `sqrt(Q_SP² + U_SP²)` |
| 4 | COUNTS | counts of 0° + 90° |
| 5 | QERR | Q error map |
| 6 | UERR | U error map |

Maps are indexed `[x_bin, y_bin]`. x goes from −7.5 to +7.5 and y from **+7.5 to −7.5**.

## Notes

* **Spurious correction.** `MakeMaps(gpd)` stores `gpd` as the correction and subtracts its map from the **raw** q/u. Calling it again never subtracts twice. Cuts (`CutUnderPeak`, `EllipticCut`, ...) and `RebinAll` keep the correction. `ResetSpurious()` removes it. `GetModulation()`/`GetPsi()` are always the uncorrected totals.
* Every event must fall inside the ±7.5 mm map. With `sCut=False`, events outside it raise a clear `ValueError` (use `CutTheEdges()`).
* The PRIMARY HDU of `SpMap*.fits` stores `μ ± 1σ` of the PHA peak, while the events used are within `μ ± 3σ` (`CutUnderPeak` default).
