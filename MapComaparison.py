import argparse
import numpy as np
from matplotlib import pyplot as plt
from XCFGraph import GFigure
from XCFgpd import GPD

parser = argparse.ArgumentParser()
parser.add_argument("--map1", type=str, help = "Path to first map", required=True)
parser.add_argument("--map2", type=str, help = "Path to second map", required=True)
args = parser.parse_args()

gpd1 = GPD()
gpd1.ReadSpurious(args.map1)
q1 = gpd1.GetQ()
u1 = gpd1.GetU()
q1_err = gpd1.GetQerr()
u1_err = gpd1.GetUerr()
gpd2 = GPD()
gpd2.ReadSpurious(args.map2)
q2 = gpd2.GetQ()
u2 = gpd2.GetU()
q2_err = gpd2.GetQerr()
u2_err = gpd2.GetUerr()
M1 = np.sqrt(q1**2 + u1**2)
M2 = np.sqrt(q2**2 + u2**2)

r_q = np.abs(q1/q2)
r_u = np.abs(u1/u2)

RCanvas = GFigure(1,2)
qm = RCanvas.AddMap(r_q, vmin=0., vmax=2., lim=7.5, position=0)
RCanvas.AddColorBar(qm, 'Qlow/Qhigh', position=0)
RCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
RCanvas.SetTitle(r'$Q_{low}\, vs\,Q_{high}$', position=0)

um = RCanvas.AddMap(r_u, vmin=0., vmax=2., lim=7.5, position=1)
RCanvas.AddColorBar(um, 'Ulow/Uhigh', position=1)
RCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=1)
RCanvas.SetTitle(r'$U_{low}\, vs\, U_{high}$', position=1)

r_qerr = np.sqrt( (q1_err/q2)**2 + (q1*q2_err/(q2**2))**2 )
r_uerr = np.sqrt( (u1_err/u2)**2 + (u1*u2_err/(u2**2))**2 )
Zq = np.abs(r_q - 1)/r_qerr
Zu = np.abs(r_u - 1)/r_uerr

DCanvas = GFigure(1,2)
q1.reshape(300*300)
q2.reshape(300*300)
u1.reshape(300*300)
u2.reshape(300*300)
DCanvas.AddHist(q1, 200, [-1., 1.], label='low rate Q', normalized=True, position=0)
DCanvas.AddHist(q2, 200, [-1., 1.], label='high rate Q', normalized=True, position=0)
DCanvas.AddHist(u1, 200, [-1., 1.], label='low rate U', normalized=True, position=1)
DCanvas.AddHist(u2, 200, [-1., 1.], label='high rate U', normalized=True, position=1)
DCanvas.AxesLabels(xlabel='Q', ylabel='Normalized Entries', position=0)
DCanvas.AxesLabels(xlabel='U', ylabel='Normalized Entries', position=1)
DCanvas.SetTitle('Q distribution', position=0)
DCanvas.SetTitle('U distribution', position=1)
DCanvas.TurnOnLegend(position=0)
DCanvas.TurnOnLegend(position=1)

ZCanvas = GFigure(1,2)
qz = ZCanvas.AddMap(Zq, vmin=0., vmax=1.96, lim=7.5, position=0)
ZCanvas.AddColorBar(qz, 'Z', position=0)
ZCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
ZCanvas.SetTitle(r'$Q_{low}\, vs\,Q_{high}$ compatibility', position=0)

uz = ZCanvas.AddMap(Zu, vmin=0., vmax=1.96, lim=7.5, position=1)
ZCanvas.AddColorBar(uz, 'Z', position=1)
ZCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=1)
ZCanvas.SetTitle(r'$U_{low}\, vs\, U_{high}$ compatibility', position=1)

print(f'Compatible Q bins: {len(Zq[np.where(Zq>1.96)])}')
print(f'Compatible U bins: {len(Zu[np.where(Zu>1.96)])}')

ModulationCanvas = GFigure(1, 2)
mapM = ModulationCanvas.AddMap(M1/M2, vmin=0., vmax=2, lim=7.5, position=0)
ModulationCanvas.AddColorBar(mapM, r'$M_{low}/M_{high}$', position=0)
ModulationCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
ModulationCanvas.SetTitle(r'$M_{low} vs M_{high}$', position=0)

M1_cutted = ModulationCanvas.EdgeCut(M1, 7)
M2_cutted = ModulationCanvas.EdgeCut(M2, 7)
ModulationCanvas.AddHist(M1_cutted, 150, [0., 2.], label='Low rate Modulation', normalized=True, position=1)
ModulationCanvas.AddHist(M2_cutted, 150, [0., 2.], label='High rate Modulation', normalized=True, position=1)
ModulationCanvas.AxesLabels(xlabel='M', ylabel='Entries', position=1)
ModulationCanvas.SetTitle('M distribution', position=1)
ModulationCanvas.TurnOnLegend(position=1)
ModulationCanvas.AxesLimits([0,2.], [None, None], 1)
RCanvas.GDraw()


