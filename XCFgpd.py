"""GPD detector geometry, spurious modulation maps and single track viewer."""
import argparse

import astropy.io.fits as pf
import numpy as np


class GPD():
    """Detector description: map binning, pixel pitch and (optional) spurious Q/U maps."""

    def __init__(self, GPD_number=35, mapbins=300):
        """Creates a GPD with mapbins x mapbins map and zero spurious modulation."""
        self.GPD_number = GPD_number
        self.nRows = mapbins
        self.nColumns = mapbins
        self.OrzPitch = 0.050    # mm
        self.VertPitch = 0.0433  # mm
        self.AxLimit = 7.5       # mm
        x_, y_, _, _ = self.MapLimits()
        self.qSpurious = np.zeros((x_, y_))
        self.uSpurious = np.zeros((x_, y_))
        self.qSpErr = np.zeros((x_, y_))
        self.uSpErr = np.zeros((x_, y_))

    def GetRows(self):
        """Returns the number of map rows."""
        return self.nRows

    def GetColumns(self):
        """Returns the number of map columns."""
        return self.nColumns

    def GetGPDNumber(self):
        """Returns the GPD serial number."""
        return self.GPD_number

    def GetLimits(self):
        """Returns the half size of the active area [mm]."""
        return self.AxLimit

    def GetOrzPitch(self):
        """Returns the horizontal pixel pitch [mm]."""
        return self.OrzPitch

    def GetQ(self):
        """Returns the spurious Q map."""
        return self.qSpurious

    def GetQerr(self):
        """Returns the spurious Q error map."""
        return self.qSpErr

    def GetU(self):
        """Returns the spurious U map."""
        return self.uSpurious

    def GetUerr(self):
        """Returns the spurious U error map."""
        return self.uSpErr

    def GetVertPitch(self):
        """Returns the vertical pixel pitch [mm]."""
        return self.VertPitch

    def SetSpuriousMap(self, qMap, uMap):
        """Sets the spurious Q and U maps."""
        self.qSpurious = qMap
        self.uSpurious = uMap

    def SetSPErrorMap(self, qErr, uErr):
        """Sets the spurious Q and U error maps."""
        self.qSpErr = qErr
        self.uSpErr = uErr

    def MapLimits(self):
        """Returns (nX, nY, x bin edges, y bin edges); y edges go from +lim to -lim."""
        x_range = np.linspace(-self.AxLimit, self.AxLimit, self.nColumns+1)
        y_range = np.linspace(self.AxLimit, -self.AxLimit, self.nRows+1)
        return len(x_range)-1, len(y_range)-1, x_range, y_range

    def ReadSpurious(self, filePath, energy=2.28):
        """Loads Q, U and their errors from a spurious map file made by XCFspurious.py (energy is not used yet)."""
        with pf.open(filePath) as data_map:
            q = np.array(data_map[1].data)     # Q_SP
            u = np.array(data_map[2].data)     # U_SP
            qerr = np.array(data_map[5].data)  # QERR
            uerr = np.array(data_map[6].data)  # UERR
        if q.shape != (self.nColumns, self.nRows):
            print(f'WARNING: spurious map {filePath} is {q.shape}, but this GPD uses {self.nColumns}x{self.nRows} bins')
        self.SetSpuriousMap(q, u)
        self.SetSPErrorMap(qerr, uerr)


class Track(GPD):
    """Reads an event file to draw the pixel tracks of single events."""

    def __init__(self, file_path, GPD_number=35):
        """Loads the EVENTS table of file_path."""
        super().__init__(GPD_number)
        self._hdul = pf.open(file_path)  # kept open: events are read lazily
        self.events = self._hdul['EVENTS'].data

    def _ij2xy(self, ij):
        """Converts (row j, column i) pixel indexes to absolute (y, x) in mm."""
        ij = np.asarray(ij)
        j, i = ij[..., 0], ij[..., 1]
        y = self.AxLimit - j*self.VertPitch
        x = np.where(j % 2 == 0, (i+0.5)*self.OrzPitch, i*self.OrzPitch) - self.AxLimit
        return np.stack((y, x), axis=-1)

    def ViewTrack(self, n):
        """Draws the hexagonal pixels of event n coloured by their charge."""
        from XCFGraph import GFigure  # imported here to avoid a circular import
        ev = self.events[n]
        min_col, max_col, min_row, max_row = ev['MIN_CHIPX'], ev['MAX_CHIPX'], ev['MIN_CHIPY'], ev['MAX_CHIPY']
        Nrows = max_row-min_row
        Ncols = max_col-min_col

        j_, i_ = np.meshgrid(np.arange(min_row, max_row+1), np.arange(min_col, max_col+1), indexing='ij')
        xy = self._ij2xy(np.column_stack((j_.ravel(), i_.ravel())))
        y = xy[:, 0]
        x = xy[:, 1]

        # Only main track will be considered
        ch = ev['PIX_PHAS_EQ']
        myCanvas = GFigure(1, 1, figsize=(Ncols*0.92, Nrows*0.81))
        myCanvas.SetTitle(f"Evnt number {n}. Baricenter: {ev['ABSX']:.3} mm, {ev['ABSY']:.3} mm")
        myCanvas.AddScatter(x, y, ch, 'h', 800, 'Event track')
        myCanvas.AddScatter(ev['BARX'], ev['BARY'], marker='*', s=600, label='Charge Baricenter')
        myCanvas.SetEqual()
        myCanvas.AxesLabels('X [mm]', 'Y [mm]')
        myCanvas.TurnOnLegend()
        myCanvas.GDraw()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw the track of one event.')
    parser.add_argument('file', help='FITS file with an EVENTS table (with pixel info)')
    parser.add_argument('--event', type=int, default=0, help='Event number to draw (default 0)')
    args = parser.parse_args()
    Track(args.file).ViewTrack(args.event)
