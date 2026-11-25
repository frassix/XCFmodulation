"""Quick diagnostic of one run: standard maps, Q/U maps and q/u distributions before and after the 7 mm edge cut."""
import argparse

from scipy.ndimage import gaussian_filter

from XCFGraph import GFigure, GPDMaps
from XCFreader import GRun

parser = argparse.ArgumentParser(description='Diagnostic plots of one run.')
parser.add_argument("--inFile", type=str, help="Path to fits file or directory", required=True)
args = parser.parse_args()

Run = GRun(args.inFile)
StdMaps = GPDMaps(Run)
StdMaps(True, f'{args.inFile} maps')

Q, U = Run.GetQMap(), Run.GetUMap()
q, u = Run.GetQ(), Run.GetU()
Run.CutTheEdges()
Q_cutted, U_cutted = Run.GetQMap(), Run.GetUMap()
q_cutted, u_cutted = Run.GetQ(), Run.GetU()

QUCanvas = GFigure(1, 2)
for position, (name, full, cutted) in enumerate((('Q', Q, Q_cutted), ('U', U, U_cutted))):
    mapId = QUCanvas.AddMap(gaussian_filter(full, sigma=1), vmin=-2., vmax=2., lim=7.5, position=position)
    QUCanvas.AddMap(gaussian_filter(cutted, sigma=1), vmin=-2., vmax=2., lim=7., position=position)
    QUCanvas.AddColorBar(mapId, name, position=position)
    QUCanvas.AxesLabels(xlabel='X [mm]', ylabel='Y [mm]', position=position)
    QUCanvas.SetTitle(rf'${name}_{{sp}}$', position=position)

QCanvas = GFigure(1, 2)
for position, (name, full, cutted) in enumerate((('q', q, q_cutted), ('u', u, u_cutted))):
    QCanvas.AddHist(full, 200, [-2., 2.], label=rf'${name}_i$ [-7.5,7.5]', position=position)
    QCanvas.AddHist(cutted, 200, [-2., 2.], label=rf'${name}_i$ [-7,7]', position=position)
    QCanvas.AxesLabels(xlabel=rf'${name}_{{i}}$', ylabel='counts', position=position)
    QCanvas.SetTitle(rf'${name}_{{i}}$', position=position)
    QCanvas.TurnOnLegend(position)
QUCanvas.GDraw()
