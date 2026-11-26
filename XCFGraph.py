"""Plotting helpers built on matplotlib: a generic figure (GFigure) and the standard run plots (GPDMaps)."""
import numpy as np
from cycler import cycler
from matplotlib import pyplot as plt
from matplotlib.text import Text

import XCFgpd as gpd


class GFigure():
    """Figure with a grid of axes; every Add*/Set* method acts on the axes at index `position`."""

    def __init__(self, nRows=1, nColumns=1, shareX=False, shareY=False, figsize=None, fontsize=16):
        """Creates the figure with nRows x nColumns axes; figsize=None picks a size that fits the number of panels."""
        plt.ion()
        self.xShape, self.yShape, _, _ = gpd.GPD().MapLimits()
        self.DotSize = 12
        self.LegendSize = fontsize+2
        self.LineWidth = 2
        self.ThicksSize = fontsize
        self.FontSize = fontsize
        self.TitleSize = fontsize+2
        self.cmap = 'viridis'
        self.contour_cmap = 'YlOrRd'
        self.palette = ['#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a']
        self.ResetAround()
        if figsize is None:
            figsize = (min(5.5*nColumns, 18), min(5*nRows, 10))
        self._NewFigure(nRows, nColumns, shareX, shareY, figsize)

    def _NewFigure(self, nRows, nColumns, shareX, shareY, figsize):
        """Creates the matplotlib figure with automatic layout and text that follows the window size."""
        # 'constrained' layout re-arranges titles, labels and colorbars at every redraw, so they never overlap
        self.figure, ax = plt.subplots(nrows=nRows, ncols=nColumns, sharex=shareX, sharey=shareY,
                                       figsize=figsize, layout='constrained')
        self.axes = np.atleast_1d(ax).ravel()
        self._refSize = self.figure.get_size_inches().copy()
        self._baseFont = {}
        # a lambda (not a bound method) keeps the callback alive after this GFigure object is gone
        self.figure.canvas.mpl_connect('resize_event', lambda event: self._OnResize(event))

    def _OnResize(self, event=None):
        """Shrinks every text when the window is smaller than the figure size (never enlarges it)."""
        w, h = self.figure.get_size_inches()
        scale = min(1., w/self._refSize[0], h/self._refSize[1])
        tickLabels = set()
        for ax in self.figure.axes:  # colorbars included
            tickLabels.update(ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both'))
            ax.tick_params(labelsize=self.ThicksSize*scale)
        for text in self.figure.findobj(Text):
            if text not in tickLabels:
                text.set_fontsize(self._baseFont.setdefault(text, text.get_fontsize())*scale)

    # ---------------------------------------------------------------- style
    def ResetAround(self):
        """Applies palette and tick size to matplotlib rc params."""
        plt.rc('axes', prop_cycle=cycler('color', self.palette), linewidth=1)
        plt.rc('xtick', labelsize=self.ThicksSize)
        plt.rc('ytick', labelsize=self.ThicksSize)

    def SetColormap(self, cm):
        """Sets the colormap used by maps and scatter plots."""
        self.cmap = cm

    def SetContourColors(self, cc):
        """Sets the colormap used by contour plots."""
        self.contour_cmap = cc

    def SetLegendSize(self, ls):
        """Sets the legend font size."""
        self.LegendSize = ls

    def SetTickSize(self, ts):
        """Sets the tick label size."""
        self.ThicksSize = ts
        self.ResetAround()

    def SetFontSize(self, fs):
        """Sets the axis label font size."""
        self.FontSize = fs

    def SetPalette(self, palette):
        """Sets the list of line colours."""
        self.palette = palette
        self.ResetAround()

    def SetTitleSize(self, ts):
        """Sets the title font size."""
        self.TitleSize = ts

    # ---------------------------------------------------------------- drawing
    def AddColorBar(self, mapId, label, position=0):
        """Adds a colorbar for mapId, as tall as the map."""
        cax = self.axes[position].inset_axes([1.04, 0., 0.05, 1.])  # follows the map box, also with equal aspect
        cbar = self.figure.colorbar(mapId, cax=cax)
        cbar.set_label(label=label, fontsize=self.FontSize, weight='bold')

    def AddContour(self, x, y, z, levels=3, alpha=1, position=0):
        """Adds a contour plot of z(x, y)."""
        self.axes[position].contour(x, y, z, levels=levels, alpha=alpha, linewidths=self.LineWidth, cmap=self.contour_cmap)

    def AddErrorbar(self, x, y, sigmaX=0., sigmaY=0., fmt='.', label='ScatterPlot', position=0):
        """Adds points with error bars."""
        return self.axes[position].errorbar(x, y, xerr=sigmaX, yerr=sigmaY, markersize=self.DotSize, markeredgecolor='black',
                                            linewidth=self.LineWidth, label=label, fmt=fmt)

    def AddExternalHist(self, binsedges, weights, alpha=0.6, label='ExtHist', position=0):
        """Draws an already computed histogram (bin edges + contents)."""
        edgecolor = 'None' if len(weights) > 100 else 'black'
        return self.axes[position].hist(binsedges[:-1], bins=binsedges, weights=weights, label=label, alpha=alpha,
                                        edgecolor=edgecolor, linewidth=1)

    def AddFunctionPlot(self, xLimits, function, parameters, label='Function', sigma=None, chi2=0, position=0):
        """Draws function(x, *parameters) in xLimits with a box showing chi2 and parameters."""
        X = np.linspace(xLimits[0], xLimits[1], 200)
        Y = function(X, *parameters)
        plot = self.AddPlot(X, Y, '-', label=label, position=position)
        info = rf"$\chi^2$ = {chi2:.1f}" + "\n"
        for i, par in enumerate(parameters):
            if sigma is None:
                info += f"Par{i}: {par:.2f}\n"
            else:
                info += f"Par{i}: {par:.2f} ± {sigma[i]:.2f}\n"
        # x in data units (right of the curve), y at 70% of the axes height: the box never ends up off-screen
        ax = self.axes[position]
        ax.text(1.1*xLimits[1], 0.7, info, transform=ax.get_xaxis_transform(), fontsize=self.LegendSize-4,
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle="round,pad=0.3"))
        return plot

    def AddHist(self, x, nBins, limits, alpha=0.8, label='Histogram', normalized=False, position=0):
        """Histograms x and draws it."""
        h, b = np.histogram(x, bins=nBins, range=limits, density=normalized)
        return self.AddExternalHist(b, h, alpha=alpha, label=label, position=position)

    def AddMap(self, xy, vmax=0.3, vmin=-0.3, lim=7.5, position=0):
        """Draws a 2D map xy[x, y] over [-lim, lim] x [-lim, lim]."""
        return self.axes[position].imshow(xy.T, vmax=vmax, vmin=vmin, extent=[-lim, lim, -lim, lim],
                                          interpolation='nearest', cmap=self.cmap)

    def AddPlot(self, x, y, fmt='.', label='Plot', position=0):
        """Adds a line/marker plot."""
        return self.axes[position].plot(x, y, fmt, markersize=self.DotSize, markeredgecolor='black',
                                        linewidth=self.LineWidth, label=label)

    def AddScatter(self, x, y, z=None, marker='.', s=12, label='Scatter', edgecolor='black', position=0):
        """Adds a scatter plot, optionally coloured by z."""
        return self.axes[position].scatter(x, y, c=z, cmap=self.cmap, marker=marker, edgecolor=edgecolor, s=s, label=label)

    def DrawEllipse(self, xc, yc, a, b, angle=0, label='', position=0):
        """Draws an ellipse centred in (xc, yc), semi-axes a, b, rotated by angle [deg]."""
        angle = np.radians(angle)
        theta = np.linspace(0, 2 * np.pi, 360)
        x = a * np.cos(theta)
        y = b * np.sin(theta)
        x_rot = xc + x * np.cos(angle) - y * np.sin(angle)
        y_rot = yc + x * np.sin(angle) + y * np.cos(angle)
        self.AddPlot(x_rot, y_rot, '-', label=label, position=position)

    def DrawVline(self, x0, y0, y1, label='', position=0):
        """Draws a vertical line at x0 from y0 to y1."""
        self.axes[position].vlines(x0, y0, y1, colors='m', label=label)

    # ---------------------------------------------------------------- axes
    def ApplyTightLayout(self, pad=1.08, h_pad=None, w_pad=None, rect=None):
        """Applies matplotlib tight_layout once (replaces the automatic layout; call it before adding colorbars)."""
        self.figure.tight_layout(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

    def AxesLabels(self, xlabel, ylabel, weight='bold', position=0):
        """Sets x and y labels."""
        self.axes[position].set_xlabel(xlabel, fontsize=self.FontSize, fontweight=weight)
        self.axes[position].set_ylabel(ylabel, fontsize=self.FontSize, fontweight=weight)

    def AxesLimits(self, xlim=(None, None), ylim=(None, None), position=0):
        """Sets x and y limits (None leaves a limit automatic)."""
        self.axes[position].set_xlim(left=xlim[0], right=xlim[1])
        self.axes[position].set_ylim(bottom=ylim[0], top=ylim[1])

    def AxesTicks(self, xticks=None, yticks=None, xlabels=None, ylabels=None, position=0):
        """Sets tick positions and (optional) tick labels."""
        if xticks is not None:
            self.axes[position].set_xticks(xticks, xlabels)
        if yticks is not None:
            self.axes[position].set_yticks(yticks, ylabels)

    def EdgeCut(self, Z, edge=7):
        """Returns the flattened part of map Z inside [-edge, edge] x [-edge, edge]."""
        lim = 7.5
        dx = (2 * lim) / self.xShape
        dy = (2 * lim) / self.yShape
        x_start_idx = int((lim - edge) / dx)
        x_end_idx = int((lim + edge) / dx)
        y_start_idx = int((lim - edge) / dy)
        y_end_idx = int((lim + edge) / dy)
        return Z[x_start_idx:x_end_idx, y_start_idx:y_end_idx].flatten()

    def SetEqual(self, position=0):
        """Sets equal aspect ratio."""
        self.axes[position].set_aspect('equal', adjustable='datalim')

    def SetFigureTitle(self, suptitle):
        """Sets the figure title."""
        self.figure.suptitle(suptitle, fontsize=self.TitleSize, weight='bold')

    def SetTitle(self, title, weight='bold', position=0):
        """Sets the axes title."""
        self.axes[position].set_title(title, fontsize=self.TitleSize, fontweight=weight)

    def TurnOnGrid(self, position=0):
        """Shows the grid."""
        self.axes[position].grid()

    def TurnOnLegend(self, position=0):
        """Shows the legend."""
        self.axes[position].legend(fontsize=self.LegendSize)

    # ---------------------------------------------------------------- output
    def GDraw(self):
        """Shows all open figures and blocks until they are closed."""
        plt.show(block=True)

    def GClose(self):
        """Closes this figure."""
        plt.close(self.figure)

    def Save(self, fileName, dpi=150):
        """Saves this figure to fileName (format from the extension: png, pdf, ...)."""
        self.figure.savefig(fileName, dpi=dpi)


class GPDMaps(GFigure):
    """Standard 3-panel view of a run: counts map, PHA distribution and phi distribution (+ optional projections)."""

    def __init__(self, data, figsize=None):
        """Creates the 1x3 figure for the run `data` (a GRun)."""
        super().__init__(1, 3, figsize=figsize)
        self.figsize = figsize
        self.data = data

    def __call__(self, projections=True, suptitle='title'):
        """Draws the three panels and, if projections, the X/Y projections figure."""
        self.SetFigureTitle(suptitle=suptitle)
        self.CountMap()
        self.PhaHist()
        self.PhiHist()
        if projections:
            self.MakeProjections(suptitle=suptitle)

    def ContourCounts(self, position=0):
        """Overlays the contour of the counts map."""
        c = self.data.GetCounts()
        X, Y = np.meshgrid(np.linspace(-7.5, 7.5, self.xShape), np.linspace(7.5, -7.5, self.yShape))
        self.AddContour(X, Y, c.T, position=position)

    def CountMap(self):
        """Draws the counts map in panel 0."""
        c = self.data.GetCounts()
        mapId = self.AddMap(c, vmax=c.max(), vmin=0., lim=7.5, position=0)
        self.AddColorBar(mapId, 'Counts', 0)
        self.AxesLabels('AbsX [mm]', 'AbsY [mm]', position=0)
        self.SetTitle('Counts Map', position=0)

    def GetProjection(self):
        """Returns the projections figure."""
        return self.ProjCanvas

    def MakeProjections(self, nBins=300, limits=(-7.5, 7.5), suptitle=''):
        """Creates a second figure with the X and Y projections."""
        self.ProjCanvas = GFigure(1, 2)
        self.ProjCanvas.SetFigureTitle(suptitle=suptitle)
        for position, values, axis in ((0, self.data.GetAbsX(), 'X'), (1, self.data.GetAbsY(), 'Y')):
            self.ProjCanvas.AddHist(values, nBins, limits, label=f'Abs{axis}', position=position)
            self.ProjCanvas.AxesLabels(f'{axis} [mm]', 'Entries', position=position)
            self.ProjCanvas.SetTitle(f'{axis} projections', position=position)
            self.ProjCanvas.TurnOnGrid(position)

    def PhaHist(self, nBins=200, limits=(0, 30000)):
        """Draws the normalized PHA distribution in panel 1."""
        self.AddHist(self.data.GetPha(), nBins, limits, label='Pha', normalized=True, position=1)
        self.AxesLabels(xlabel='Pha', ylabel='Normalized Counts', position=1)
        self.SetTitle('Pha Distribution', position=1)
        self.AxesLimits([0, None], position=1)
        self.TurnOnGrid(1)
        self.TurnOnLegend(1)

    def PhiHist(self, nBins=200, limits=(-180., 180.)):
        """Draws the phi distribution in panel 2, with modulation and Psi in the title."""
        self.AddHist(np.degrees(self.data.GetPhi()), nBins, limits, label='Phi2', position=2)
        self.AxesLabels(xlabel='Phi2 [deg]', ylabel='Entries', position=2)
        self.SetTitle(f'Modulation: {(self.data.GetModulation()*100):.3}%\nPsi: {np.degrees(self.data.GetPsi()):.3} deg', position=2)
        self.AxesLimits([-180, 180], position=2)
        self.TurnOnGrid(2)
        self.TurnOnLegend(2)

    def UpdateData(self, data):
        """Replaces the run and redraws the panels in a new figure."""
        self.data = data
        plt.close(self.figure)
        self._NewFigure(1, 3, False, False, self._refSize)
        self.__call__()


if __name__ == '__main__':
    # Small demo of the plotting helpers
    myCanvas = GFigure(1, 2, figsize=(14, 7))
    myCanvas.SetTitle('XCF graph demo')
    myCanvas.AddPlot([0, 2, 4], [1, 2, 3], '.-', label='Plot')
    myCanvas.AddErrorbar([0, 1, 2], [0, 1, 2], 0.1, 0.2, label='Errorbar')
    myCanvas.AxesLabels('X [cm]', 'Y [cm]')
    myCanvas.TurnOnGrid()
    myCanvas.TurnOnLegend()
    rng = np.random.default_rng(0)
    myCanvas.AddHist(rng.normal(5, 2, 1000), 20, [0, 10], label='Hist', position=1)
    myCanvas.SetTitle('XCF hist demo', position=1)
    myCanvas.AxesTicks([3, 6, 9], None, ['A', 'B', 'C'], position=1)
    myCanvas.TurnOnLegend(1)
    myCanvas.GDraw()
