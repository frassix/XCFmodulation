"""Shows ASIC equalization maps (EQ_SLOPE) and their ratio."""
import argparse
import os

import astropy.io.fits as pf
import numpy as np
from matplotlib import pyplot as plt

from XCFGraph import GFigure


class EqMap():
    """Equalization map read from a single FITS file (columns CHIPX, CHIPY, EQ_SLOPE)."""

    def __init__(self, file_path):
        """Reads the equalization file."""
        self.path = file_path
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"Given path '{self.path}' is not a file (only single files can be read)")
        print("\n" + "-"*100 + "\n" + "-"*100 + f"\nReading single file: {self.path}")
        self.ReadOneFile()
        print('Done reading\n-------------------------------------')

    def DivideBy(self, Den):
        """Divides the slopes by Den (array or EqMap), reshaped to the same shape."""
        if isinstance(Den, EqMap):
            Den = Den.GetE()
        self.e = self.e/np.reshape(Den, np.shape(self.e))

    def GetE(self):
        """Returns the EQ_SLOPE values."""
        return self.e

    def PlotMap(self, show=False):
        """Plots the slope map (300x352 pixels) and its distribution."""
        Fig = GFigure(1, 2)
        plot = Fig.AddMap(self.e.reshape(300, 352), vmax=1.5, vmin=0.)
        Fig.SetFigureTitle(f'{self.path}')
        Fig.SetTitle('ASIC Equalization Map')
        Fig.AxesLabels('X [mm]', 'Y [mm]')
        Fig.AddColorBar(plot, 'Eq Slope')
        Fig.AddHist(self.e, 100, limits=[0., 2], position=1)
        Fig.SetTitle('Eq Slope Distribution', position=1)
        Fig.AxesLabels('Eq Slope', 'Entries [#]', position=1)
        Fig.TurnOnGrid(1)
        if show:
            Fig.GDraw()

    def ReadOneFile(self):
        """Loads CHIPX, CHIPY and EQ_SLOPE from the first extension."""
        with pf.open(self.path) as data_f:
            df = data_f[1].data
            self.x = np.array(df['CHIPX'])
            self.y = np.array(df['CHIPY'])
            self.e = np.array(df['EQ_SLOPE'])
        print(f'Min Eq_Slope value: {self.e.min()}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show equalization maps.')
    parser.add_argument('files', nargs='*', help='Equalization FITS files to show')
    parser.add_argument('--ratio', nargs=2, metavar=('NUM', 'DEN'), help='Also show the map NUM divided by DEN')
    args = parser.parse_args()
    if not args.files and not args.ratio:
        parser.error('give at least one file or --ratio NUM DEN')

    for path in args.files:
        EqMap(path).PlotMap()
    if args.ratio:
        Ratio = EqMap(args.ratio[0])
        Ratio.DivideBy(EqMap(args.ratio[1]))
        Ratio.path = f'{args.ratio[0]} / {args.ratio[1]}'
        Ratio.PlotMap()
    plt.show(block=True)
