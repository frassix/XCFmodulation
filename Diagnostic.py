import argparse
import numpy as np

from matplotlib import pyplot as plt
from XCFreader import GRun
from XCFGraph import GFigure, GPDMaps
from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser()
parser.add_argument("--inFile", type=str, help = "Path to fits files", required=True)
args = parser.parse_args()

Run = GRun(args.inFile)
StdMaps = GPDMaps(Run)
StdMaps(True, f'{args.inFile} maps')

Q = Run.GetQMap()
U = Run.GetUMap()
q = Run.GetQ()
u = Run.GetU()
Run.CutTheEdges()
Q_cutted = Run.GetQMap()
U_cutted = Run.GetUMap()
q_cutted = Run.GetQ()
u_cutted = Run.GetU()

QUCanvas = GFigure(1,2)
mapQ = QUCanvas.AddMap(gaussian_filter(Q, sigma=1), vmin=-2., vmax=2., lim=7.5, position=0)
QUCanvas.AddMap(gaussian_filter(Q_cutted, sigma=1), vmin=-2., vmax=2., lim=7., position=0)
QUCanvas.AddColorBar(mapQ, 'Q', position=0)
QUCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=0)
QUCanvas.SetTitle(r'$Q_{sp}$', position=0)
mapU = QUCanvas.AddMap(gaussian_filter(U, sigma=1), vmin=-2., vmax=2., lim=7.5, position=1)
QUCanvas.AddMap(gaussian_filter(Q_cutted, sigma=1), vmin=-2., vmax=2., lim=7., position=1)
QUCanvas.AddColorBar(mapU, 'U', position=1)
QUCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=1)
QUCanvas.SetTitle(r'$U_{sp}$', position=1)

QCanvas = GFigure(1,2)
QCanvas.AddHist(q, 200, [-2.,2.], label=r'$q_i$ [-7.5,7.5]', position=0)
QCanvas.AddHist(q_cutted, 200, [-2., 2.], label=r'$q_i$ [-7,7]', position=0)
QCanvas.AxesLabels(xlabel=r'$q_{i}$', ylabel='counts', position=0)
QCanvas.SetTitle(r'$q_{i}$', position=0)
QCanvas.AddHist(u, 200, [-2.,2.], label=r'$u_i$ [-7.5,7.5]', position=1)
QCanvas.AddHist(u_cutted, 200, [-2., 2.], label=r'$u_i$ [-7,7]', position=1)
QCanvas.AxesLabels(xlabel=r'$u_{i}$', ylabel='counts', position=1)
QCanvas.SetTitle(r'$u_{i}$', position=1)
QCanvas.TurnOnLegend(0)
QCanvas.TurnOnLegend(1)
QUCanvas.GDraw()
