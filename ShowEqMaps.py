"""ASIC equalization maps (EQ_SLOPE per pixel): map, distribution, column/row projections and ratio of two maps."""
import argparse
import os

import astropy.io.fits as pf
import numpy as np
from matplotlib import pyplot as plt

from XCFGraph import GFigure
from XCFgpd import GPD
from XCFreader import PrintFitsHeaders


class EqMap():
    """Equalization map read from a single FITS file (extension PIXEQ, columns CHIPX, CHIPY, EQ_SLOPE)."""

    def __init__(self, file_path, GPD_number=35):
        """Reads the equalization file; pixel pitches come from the GPD geometry."""
        self.path = file_path
        self.gpd = GPD(GPD_number)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Given path '{self.path}' is not a file (only single files can be read)")
        print("\n" + "-"*100 + "\n" + "-"*100 + f"\nReading single file: {self.path}")
        self.ReadOneFile()
        print('Done reading\n-------------------------------------')

    def DivideBy(self, Den):
        """Divides the slopes by Den: an EqMap (matched pixel by pixel) or an array with the same number of pixels."""
        if isinstance(Den, EqMap):
            self.e = self.e/Den.Map2D()[self.x, self.y]
        else:
            self.e = self.e/np.reshape(Den, np.shape(self.e))

    def Extent(self):
        """Returns [x0, x1, y0, y1] of the map in detector coordinates [mm] (same frame as ABSX/ABSY)."""
        nx, ny = self.Shape()
        lim, P, V = self.gpd.GetLimits(), self.gpd.GetOrzPitch(), self.gpd.GetVertPitch()
        return [-lim, -lim + nx*P, lim - (ny-0.5)*V, lim + 0.5*V]

    def GetE(self):
        """Returns the EQ_SLOPE values (one per row of the file)."""
        return self.e

    def Map2D(self):
        """Returns the slopes as a 2D array [column CHIPX, row CHIPY] (NaN for missing pixels)."""
        Z = np.full(self.Shape(), np.nan)
        Z[self.x, self.y] = self.e
        return Z

    def PlotMap(self, show=False, nBins=100, limits=(0., 2.)):
        """Plots the slope map and its distribution."""
        Fig = GFigure(1, 2)
        plot = Fig.AddMap(self.Map2D(), vmax=1.5, vmin=0., extent=self.Extent())
        Fig.SetFigureTitle(self.Name())
        Fig.SetTitle('ASIC Equalization Map')
        Fig.AxesLabels('X [mm]', 'Y [mm]')
        Fig.AddColorBar(plot, 'Eq Slope')
        Fig.AddHist(self.e, nBins, limits=limits, label='Eq Slope', position=1)
        Fig.SetTitle('Eq Slope Distribution', position=1)
        Fig.AxesLabels('Eq Slope', 'Entries [#]', position=1)
        Fig.TurnOnGrid(1)
        if show:
            Fig.GDraw()
        return Fig

    def PlotProjections(self, show=False):
        """Plots the mean slope of every pixel column (vs X) and of every pixel row (vs Y)."""
        Z = self.Map2D()
        x0, x1, y0, y1 = self.Extent()
        nx, ny = self.Shape()
        Fig = GFigure(1, 2)
        Fig.SetFigureTitle(self.Name())
        # rows (CHIPY) go from the top (y1) down, so their means are reversed to follow increasing Y
        for position, (edges, mean, axis, what) in enumerate((
                (np.linspace(x0, x1, nx+1), np.nanmean(Z, axis=1), 'X', 'Column'),
                (np.linspace(y0, y1, ny+1), np.nanmean(Z, axis=0)[::-1], 'Y', 'Row'))):
            Fig.AddExternalHist(edges, mean, label=f'{what} mean', position=position)
            Fig.AxesLabels(f'{axis} [mm]', f'{what} Mean Eq', position=position)
            Fig.SetTitle(f'{axis} projection', position=position)
            Fig.TurnOnGrid(position)
        if show:
            Fig.GDraw()
        return Fig

    def Name(self):
        """Returns the file name without directories (used as figure title)."""
        return ' / '.join(os.path.basename(p) for p in self.path.split(' / '))

    def PrintHead(self):
        """Prints every header of the file."""
        PrintFitsHeaders(self.path)

    def ReadOneFile(self):
        """Loads CHIPX, CHIPY and EQ_SLOPE from the PIXEQ extension (or the first one if it has another name)."""
        with pf.open(self.path) as data_f:
            df = (data_f['PIXEQ'] if 'PIXEQ' in data_f else data_f[1]).data
            self.x = np.array(df['CHIPX'], dtype=int)
            self.y = np.array(df['CHIPY'], dtype=int)
            self.e = np.array(df['EQ_SLOPE'])
        nx, ny = self.Shape()
        print(f'Pixels: {len(self.e)} ({nx} columns x {ny} rows)')
        print(f'Min Eq_Slope value: {self.e.min()}')

    def Shape(self):
        """Returns (number of columns, number of rows) of the ASIC."""
        return self.x.max()+1, self.y.max()+1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show equalization maps.')
    parser.add_argument('files', nargs='*', help='Equalization FITS files to show')
    parser.add_argument('--f', '--infile', dest='infile', action='append', default=[],
                        help='Equalization FITS file (same as giving it as positional argument)')
    parser.add_argument('--head', action='store_true', help='Print the FITS headers')
    parser.add_argument('--projections', action='store_true', help='Also plot the column/row mean projections')
    parser.add_argument('--ratio', nargs=2, metavar=('NUM', 'DEN'), help='Also show the map NUM divided by DEN')
    args = parser.parse_args()
    files = args.files + args.infile
    if not files and not args.ratio:
        parser.error('give at least one file or --ratio NUM DEN')

    maps = []
    for path in files:
        maps.append(EqMap(path))
    if args.ratio:
        Ratio = EqMap(args.ratio[0])
        Ratio.DivideBy(EqMap(args.ratio[1]))
        Ratio.path = f'{args.ratio[0]} / {args.ratio[1]}'
        maps.append(Ratio)
    for Map in maps:
        if args.head and os.path.isfile(Map.path):
            Map.PrintHead()
        Map.PlotMap()
        if args.projections:
            Map.PlotProjections()
    plt.show(block=True)
