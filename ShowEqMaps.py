import astropy.io.fits as pf
import os
import numpy as np
import sys
from XCFGraph import GFigure

class EqMap():

    def __init__(self, file_path):
        self.path = file_path
        self.x = np.array([])
        self.y = np.array([])
        self.e = np.array([])
        self.FindFiles()

    def FindFiles(self):
        if os.path.exists(self.path):
            if os.path.isfile(self.path):
                print("\n" + "-"*100 + "\n" + "-"*100 + f"\nReading single file: {self.path}")
                self.ReadOneFile()
            elif os.path.isdir(self.path):
                print(f"ERROR: given path {self.path} is dir. Only single files can be read")
                sys.exit(1)
            print('Done reading\n-------------------------------------')
        else:
            print(f"Error: given path '{self.path}' does not exists\n***Abort***")
            sys.exit(1)

    def DivideBy(self, Den):
        if np.shape(Den)==np.shape(self.e):
            self.e = self.e/Den
        else:
            Den.reshape(np.shape(self.e))
            self.e = self.e/Den

    def GetE(self):
        return self.e

    def PlotMap(self, show=False):
        Fig = GFigure(1,2)
        plot = Fig.AddMap(self.e.reshape(300,352), vmax=1.5, vmin=0.)
        Fig.SetFigureTitle(f'{self.path}')
        Fig.SetTitle('ASIC Equalization Map')
        Fig.AxesLabels('X [mm]', 'Y [mm]')
        Fig.AddColorBar(plot, 'Eq Slope')

        Fig.AddHist(self.e, 100, limits=[0.,2], position=1)
        Fig.SetTitle('Eq Slope Distribution', position=1)
        Fig.AxesLabels('Eq Slpe', 'Entries [#]', position=1)
        Fig.TurnOnGrid(1)
        if show:
            Fig.GDraw()


    def ReadOneFile(self):
        data_f = pf.open(self.path)
        df = data_f[1].data
        self.x = df['CHIPX']
        self.y = df['CHIPY']
        self.e = df['EQ_SLOPE']
        print(f'Min Eq_Slope value: {self.e.min()}')


if __name__ == '__main__':
    # TEST 1
    path1 = '/data2/XCF/gpd_data/GPD38/Feb25/flat38_50side_noCut.fits'
    path2 = '/data2/XCF/gpd_data/GPD38/Feb25/flat38_50side_tk.fits'
    path3 = '/data2/XCF/gpd_data/GPD38/Gen25/flat38_50side.fits'
    Map1 = EqMap(path1)
    Map1.PlotMap()
    Map2 = EqMap(path2)
    Map2.PlotMap()
    Map3 = EqMap(path3)
    Map3.PlotMap()
    E3 = Map3.GetE()
    Map2.DivideBy(E3)
    Map2.PlotMap(True)