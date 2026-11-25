"""Compares two spurious maps (e.g. low and high rate): ratios, Z compatibility and Q/U/M distributions."""
import argparse

import numpy as np

from XCFGraph import GFigure
from XCFgpd import GPD

parser = argparse.ArgumentParser(description='Compare two spurious map files.')
parser.add_argument("--map1", type=str, help="Path to first map (low rate)", required=True)
parser.add_argument("--map2", type=str, help="Path to second map (high rate)", required=True)
args = parser.parse_args()

gpd1 = GPD()
gpd1.ReadSpurious(args.map1)
gpd2 = GPD()
gpd2.ReadSpurious(args.map2)
q1, u1, q1_err, u1_err = gpd1.GetQ(), gpd1.GetU(), gpd1.GetQerr(), gpd1.GetUerr()
q2, u2, q2_err, u2_err = gpd2.GetQ(), gpd2.GetU(), gpd2.GetQerr(), gpd2.GetUerr()
M1 = np.sqrt(q1**2 + u1**2)
M2 = np.sqrt(q2**2 + u2**2)

with np.errstate(divide='ignore', invalid='ignore'):
    r_q = np.abs(q1/q2)
    r_u = np.abs(u1/u2)
    r_qerr = np.sqrt((q1_err/q2)**2 + (q1*q2_err/(q2**2))**2)
    r_uerr = np.sqrt((u1_err/u2)**2 + (u1*u2_err/(u2**2))**2)
    Zq = np.abs(r_q - 1)/r_qerr
    Zu = np.abs(r_u - 1)/r_uerr
    r_M = M1/M2

RCanvas = GFigure(1, 2)
DCanvas = GFigure(1, 2)
ZCanvas = GFigure(1, 2)
for position, (S, r, Z, low, high) in enumerate((('Q', r_q, Zq, q1, q2), ('U', r_u, Zu, u1, u2))):
    mapId = RCanvas.AddMap(r, vmin=0., vmax=2., lim=7.5, position=position)
    RCanvas.AddColorBar(mapId, f'{S}low/{S}high', position=position)
    RCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=position)
    RCanvas.SetTitle(rf'${S}_{{low}}\, vs\,{S}_{{high}}$', position=position)

    DCanvas.AddHist(low, 200, [-1., 1.], label=f'low rate {S}', normalized=True, position=position)
    DCanvas.AddHist(high, 200, [-1., 1.], label=f'high rate {S}', normalized=True, position=position)
    DCanvas.AxesLabels(xlabel=S, ylabel='Normalized Entries', position=position)
    DCanvas.SetTitle(f'{S} distribution', position=position)
    DCanvas.TurnOnLegend(position=position)

    mapId = ZCanvas.AddMap(Z, vmin=0., vmax=1.96, lim=7.5, position=position)
    ZCanvas.AddColorBar(mapId, 'Z', position=position)
    ZCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=position)
    ZCanvas.SetTitle(rf'${S}_{{low}}\, vs\,{S}_{{high}}$ compatibility', position=position)

print(f'Incompatible Q bins (Z>1.96): {np.count_nonzero(Zq > 1.96)}')
print(f'Incompatible U bins (Z>1.96): {np.count_nonzero(Zu > 1.96)}')

ModulationCanvas = GFigure(1, 2)
mapM = ModulationCanvas.AddMap(r_M, vmin=0., vmax=2, lim=7.5, position=0)
ModulationCanvas.AddColorBar(mapM, r'$M_{low}/M_{high}$', position=0)
ModulationCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
ModulationCanvas.SetTitle(r'$M_{low} vs M_{high}$', position=0)
ModulationCanvas.AddHist(ModulationCanvas.EdgeCut(M1, 7), 150, [0., 2.], label='Low rate Modulation', normalized=True, position=1)
ModulationCanvas.AddHist(ModulationCanvas.EdgeCut(M2, 7), 150, [0., 2.], label='High rate Modulation', normalized=True, position=1)
ModulationCanvas.AxesLabels(xlabel='M', ylabel='Entries', position=1)
ModulationCanvas.SetTitle('M distribution', position=1)
ModulationCanvas.TurnOnLegend(position=1)
ModulationCanvas.AxesLimits([0, 2.], [None, None], 1)
RCanvas.GDraw()
