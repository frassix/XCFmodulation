import astropy.io.fits as pf
import glob
import os
import numpy as np
import sys
import XCFgpd as gpd
import XCFfit as fit

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


class GRun():

    def __init__(self, file_path, sCut = True):
        self.path = file_path
        self.Cut  = sCut
        self.totalLiveTime = 0.
        self.modulation = None
        self.Phi  = np.array([])
        self.Absx = np.array([])
        self.Absy = np.array([])
        self.Pha  = np.array([])
        self.FindFiles()
        self.CalculateQUC()
        self.MakeMaps()

    def ApplyMask(self, mask, redo):
        print(f'Selected data: {len(mask)}')
        self.Phi  = self.Phi[mask]
        self.Absx = self.Absx[mask]
        self.Absy = self.Absy[mask]
        self.Pha  = self.Pha[mask]
        if redo:
            self.CalculateQUC()
            self.MakeMaps()

    def CalculateQUError(self, q, u, N):
        sigmaQ = np.sqrt( (2-q**2)/(N-1) )
        sigmaU = np.sqrt( (2-u**2)/(N-1) )
        return sigmaQ, sigmaU

    def CalculateQUC(self):
        self.q = 2*np.cos(2*self.Phi)
        self.u = 2*np.sin(2*self.Phi)
        self.modulation = np.sqrt( (np.mean(self.q))**2 + (np.mean(self.u))**2 )
        self.Psi = 0.5*np.arctan2(np.mean(self.u), np.mean(self.q))

    def CutTheEdges(self, limX=7., limY=7.):
        mask = np.where( (np.abs(self.Absx)<limX) & (np.abs(self.Absy)<limY) )
        print(f'Cutting all events out of X:[{-limX}, {limX}], Y:[{-limY}, {limY}]')
        self.ApplyMask(mask, True)

    def CutUnderPeak(self, nSigma = 3.):
        print('-'*30 + 'CUTTING PHA UNDER PEAK' + '-'*30)
        Gaussian = fit.GFit("[0]*exp(-(x-[1])**2/(2*[2]**2))")
        x, y, par0, bounds = self.MakeGaussian(self.Pha, 200, [0, 30000])
        param, cov = Gaussian(x, y, errY = np.sqrt(y), params=par0, bounds = bounds)
        self.ApplyMask(np.where(np.abs(self.Pha-param[1])<=nSigma*param[2])[0], True)
        return param, cov, Gaussian.GetFunction(), Gaussian.GetChi2()
    
    def EllipticCut(self, xc, yc, a, b, inside=True):
        if (a==0) | (b==0):
            a=1
            b=1
            print("ERROR: invalid value for elliptic shape. Setting a=b=1")
        if inside:
            mask = np.where( ((self.Absx-xc)/a)**2 + ((self.Absy-yc)/b)**2 <= 1)
            print(f"Cutting data inside the ellipse: [(x-{xc:.2})/{a:.2}]^2 + [(y-{yc:.2})/{b:.2}]^2 <= 1")
        else:
            mask = np.where( ((self.Absx-xc)/a)**2 + ((self.Absy-yc)/b)**2 > 1)
            print(f"Cutting data outside the ellipse: [(x-{xc:.2})/{a:.2}]^2 + [(y-{yc:.2})/{b:.2}]^2 > 1")
        self.ApplyMask(mask, True)

    def FindFiles(self):
        if os.path.exists(self.path):
            if os.path.isfile(self.path):
                print("\n" + "-"*100 + "\n" + "-"*100 + f"\nReading single file: {self.path}")
                print('StdCuts applyed: PHA>0 & NUM_CLU>0 & LIVETIME>15 & TRK_SIZE>0')
                self.ReadOneFile(file = self.path)
            elif os.path.isdir(self.path):
                print("\n" + "-"*100 + "\n" + "-"*100 + f"\nReading all files in dir: {self.path}")
                print('StdCuts applyed: PHA>0 & NUM_CLU>0 & LIVETIME>15 & TRK_SIZE>0')
                self.StackAllFiles()
            
            print('Done reading\n-------------------------------------')
            self.PrintInfo()
        else:
            print(f"Error: given path '{self.path}' does not exists\n***Abort***")
            sys.exit(1)

    def FindGaussCenter(self):
        print('-'*30 + 'FINDING GAUSSIAN CENTER' + '-'*30 )
        Gaussian = fit.GFit("[0]*exp(-(x-[1])**2/(2*[2]**2))")
        print('X center position')
        bX, hX, par0_x, bounds_x = self.MakeGaussian(self.Absx, 300, [-7.5, 7.5], 1)
        param_x, cov_x = Gaussian(bX, hX, errY = np.sqrt(hX), params=par0_x, bounds = bounds_x)
        temp1 = Gaussian.GetChi2()
        print('Y center position')
        bY, hY, par0_y, bounds_y = self.MakeGaussian(self.Absy, 300, [-7.5, 7.5], 1)
        param_y, cov_y = Gaussian(bY, hY, errY = np.sqrt(hY), params=par0_y, bounds = bounds_y)
        chi = [temp1, Gaussian.GetChi2()]
        return [param_x, param_y], [cov_x, cov_y], Gaussian.GetFunction(), chi

    def FunctionCut(self, expression, param, above=False, ax='x'):
        func = fit.GF1(expression=expression)
        if (ax=='x')|(ax=='X'):
            if above:
                mask = np.where( self.Absy >= func(self.Absx, *param))
            else:
                mask = np.where( self.Absy <= func(self.Absx, *param))
        elif (ax=='y')|(ax=='Y'):
            if above:
                mask = np.where( self.Absx >= func(self.Absy, *param))
            else:
                mask = np.where( self.Absx <= func(self.Absy, *param))
        else:
            print('Invalid ax in FunctionCut: setting ax to x')
            self.FunctionCut(expression, param, above, ax='x')
        print(f'Cutting data following {expression}')
        self.ApplyMask(mask)

    def GetAbsX(self):
        return self.Absx

    def GetAbsY(self):
        return self.Absy

    def GetCounts(self):
        return self.counts

    def GetIndexes(self):
        return self.bin_indices

    def GetModulation(self):
        return self.modulation
    
    def GetPhysicsMask(self):
        mask = np.where( (np.abs(self.Absx)<7) & (np.abs(self.Absy)<7) )
        return mask[0]
    
    def GetPath(self):
        return self.path

    def GetPha(self):
        return self.Pha

    def GetPhi(self):
        return self.Phi

    def GetPsi(self):
        return self.Psi

    def GetQ(self):
        return self.q

    def GetQMap(self):
        return self.Q

    def GetU(self):
        return self.u

    def GetUMap(self):
        return self.U
        
    def LinearCut(self, min, max, ax='x'):
        if (ax=='x')|(ax=='X'):
            mask = np.where( (self.Absx>=min) & (self.Absx<=max))
            print(f'Cutting data: {min}<=AbsX<={max}')
        else:
            mask = np.where( (self.Absy>=min) & (self.Absy<=max))
            print(f'Cutting data: {min}<=AbsY<={max}')
        self.ApplyMask(mask, True)

    def MakeGaussian(self, x, nBins, range, n=2):
        h, b = np.histogram(x, bins=nBins, range=range)
        centers = b[:-1] + ( b[1] - b[0] ) / 2.
        Amp = h.max()
        mu = np.average(centers, weights=h)
        sig = np.sqrt(np.average((centers-mu)**2, weights=h))
        params = [Amp, mu, sig]
        bounds = ([Amp*0.9, mu-n*sig, 0.], [Amp*1.1, mu+n*sig, 2*n*sig])
        print(params)
        RegionOfFit = np.where(h>=h.max()/2.)
        centers = centers[RegionOfFit]
        h = h[RegionOfFit]
        return centers, h, params, bounds

    def MakeMaps(self, myGPD=None):
        print("Making Q and U maps")
        if myGPD is None:
            myGPD = gpd.GPD()
        x_, y_, x_range, y_range = myGPD.MapLimits()
        q_sp = myGPD.GetQ()
        u_sp = myGPD.GetU()
        q_sp = q_sp.reshape(x_*y_)
        u_sp = u_sp.reshape(x_*y_)
        
        x_indices = np.digitize(self.Absx, x_range)-1
        y_indices = np.digitize(self.Absy, y_range)-1
        self.bin_indices = x_indices * (len(y_range)-1) + y_indices
        self.q = self.q-q_sp[self.bin_indices]
        self.u = self.u-u_sp[self.bin_indices]
        
        q_tot = np.zeros(x_*y_)
        u_tot = np.zeros(x_*y_)
        counts = np.zeros(x_*y_)
        np.add.at(q_tot,  self.bin_indices, self.q)
        np.add.at(u_tot,  self.bin_indices, self.u)
        np.add.at(counts, self.bin_indices, 1)
        self.Q = np.where(counts>0, q_tot/counts, 0)
        self.U = np.where(counts>0, u_tot/counts, 0)
        self.counts = counts
        self.Q = self.Q.reshape(x_, y_)
        self.U = self.U.reshape(x_, y_)
        self.counts = self.counts.reshape(x_, y_)
        print("Maps Ready")

    def PrintHead(self):
        with pf.open(self.path) as hdul:
            for i, hdu in enumerate(hdul):
                print(f"\nHeader {i+1} (primario o estensione):")
                for keyword, value in hdu.header.items():
                    print(f"{keyword}: {value}")

    def PrintInfo(self):
        print(
            f'\nData from {self.path}'
            f'\nTotal number of events: {len(self.Pha)}'
            f'\nTotal live-time: {self.totalLiveTime:.5} h'
            f'\nPha Max: {self.Pha.max()}'
        )
            
    def ReadOneFile(self, file):
        data_f = pf.open(file)
        df = data_f['EVENTS']
        if self.Cut:
            mask = self.StdCut(df)
        else:
            mask = self.VoidMask(df)
        self.Phi  = np.concatenate( (self.Phi,  df.data['DETPHI2'][mask] ) )
        self.Absx = np.concatenate( (self.Absx, df.data['ABSX'][mask]    ) )
        self.Absy = np.concatenate( (self.Absy, df.data['ABSY'][mask]    ) )
        self.Pha  = np.concatenate( (self.Pha,  df.data['PHA'][mask]     ) )

    def RebinAll(self, Nbins):
        FakeGpd = gpd.GPD(mapbins=Nbins)
        self.MakeMaps(FakeGpd)

    def StackAllFiles(self):
        files = glob.glob(self.path+'*.fits')
        print(f'Reading files: {files}')
        for f in files:
            self.ReadOneFile(f)

    def StdCut(self, df):
        pha = df.data['PHA']
        clu = df.data['NUM_CLU']
        lti = df.data['LIVETIME']
        trk = df.data['TRK_SIZE']
        x   = df.data['ABSX']
        y   = df.data['ABSY']
        #phi = df.data['DETPHI2']
        self.totalLiveTime += (df.data['TIME'].max()-df.data['TIME'].min())/3600
        mask = np.where( (np.abs(x)<7.5) & (np.abs(y)<7.5) & (pha>0.) & (clu>0.) & (lti>15.) & (trk>0.)  )[0]
        return mask

    def VoidMask(self, df):
        pha = df.data['PHA']
        mask = np.where(pha>=pha.min())
        return mask


class GCanvas():

    def __init__(self, fontSize=14):
        self.fontSize = fontSize
        self.StandardGPD = gpd.GPD()
        self.MapLimits()

    def AddColorMap(self, position, mapId, label):
        self.figure.colorbar(mapId, ax=self.ax[position], label=label)
        
    def AddHist(self, x, nBins, limits, position, label='Histogram'):
        h, b = np.histogram(x, bins=nBins, range=limits)
        hist = self.ax[position].hist(b[:-1], bins=b, weights=h, label=label)
        return hist

    def AddMap(self, position, xy, vmax=0.3, vmin=-0.3, lim=10):
        maps = self.ax[position].imshow(xy.T,  vmax=vmax, vmin=vmin, extent=[-lim, lim, -lim, lim], interpolation='nearest')
        return maps

    def AddFunctionPlot(self, position, x, function, parameters, label, sigma=None, chi2=0):
        X = np.linspace(x.min(), x.max(), 200)
        Y = function(X, *parameters)
        plot = self.ax[position].plot(X, Y, label=label)
        info = f"$\chi^2$ = {chi2:.1f}\n"
        for i in range(len(parameters)):
            if sigma is None:
                info += f"Par{i}: {parameters[i]:.2f}\n"
            else:
                info += f"Par{i}: {parameters[i]:.2f} ± {sigma[i]:.2f}\n"
        self.ax[position].text(1.2*x.max(), 0.8*Y.max(), info, fontsize=self.fontSize-1, bbox=dict(facecolor='white', edgecolor='black', alpha=0.5, boxstyle="round,pad=0.3"))
        return plot

    def AddErrorPlot(self, position, x, y, sigmaX=0., sigmaY=0., label=''):
        sp = self.ax[position].errorbar(x, y, xerr=sigmaX, yerr=sigmaY, label=label)
        return sp

    def AxesLabels(self, position, xlabel, ylabel, weight='bold'):
        self.ax[position].set_xlabel(xlabel, fontsize=self.fontSize, fontweight=weight)
        self.ax[position].set_ylabel(ylabel, fontsize=self.fontSize, fontweight=weight)

    def AxesLimits(self, position, xlim=[None, None], ylim=[None, None]):
        self.ax[position].set_xlim(left=xlim[0], right=xlim[1])
        self.ax[position].set_ylim(bottom=ylim[0], top=ylim[1])
            
    def AxesTitle(self, position, title, weight='bold'):
        self.ax[position].set_title(title, fontsize=self.fontSize+2, fontweight=weight)

    def AxesGrid(self, position):
        self.ax[position].grid()
        
    def AxesLegend(self, position):
        self.ax[position].legend(fontsize=self.fontSize-1)

    def CountMap(self, data):
        c = data.GetCounts()
        mapId = self.AddMap(0, c.reshape(self.xShape, self.yShape), vmax = c.max(), vmin = 0., lim=7.5)
        self.AddColorMap(0, mapId, 'Counts')
        self.AxesLabels(0, xlabel='AbsX [cm]', ylabel='AbsY [cm]')
        self.AxesTitle(0, 'Counts Map')
        
    def DataMaps(self, data):
        self.figure, self.ax = plt.subplots(1, 3, figsize=(16, 9))
        self.CountMap(data)
        self.PhaHist(data)
        self.PhiHist(data)

    def MapLimits(self):
        x_, y_, _, _ = self.StandardGPD.MapLimits()
        self.xShape = x_
        self.yShape = y_

    def NewFigureAxes(self, axRows = 1, axColumns =2):
        self.figure, self.ax = plt.subplots(axRows, axColumns, figsize=(16,9))

    def PhaHist(self, data):
        pha = data.GetPha()
        _ = self.AddHist(pha, 200, limits=[0., 30000], position = 1, label='Pha')
        self.AxesLabels(1, xlabel='Pha', ylabel='')
        self.AxesTitle(1, 'Pha Spec')
        self.AxesGrid(1)
        self.AxesLegend(1)

    def PhiHist(self, data):
        phi = data.GetPhi()
        x = data.GetAbsX()
        y = data.GetAbsY()
        mask = np.where((np.abs(x)<7.) & (np.abs(y)<7.))
        _ = self.AddHist(np.degrees(phi[mask]), 200, limits=[-180., 180.], position = 2, label='Phi2')
        self.AxesLabels(2, xlabel='Phi2 [deg]', ylabel='')
        self.AxesTitle(2, f'Modulation: {(data.GetModulation()*100):.3}% , Psi: {np.degrees(data.GetPsi()):.3} deg')
        self.AxesGrid(2)
        self.AxesLegend(2)

    def Projections(self, data):
        self.figure, self.ax = plt.subplots(1, 3, figsize=(16, 9))
        self.CountMap(data)
        self.AddHist(data.GetAbsX(), 300, [-7.5, 7.5], 1, label='AbsX')
        self.AxesLabels(1, 'X [mm]', 'Entries')
        self.AxesTitle(1, 'X projections')
        self.AddHist(data.GetAbsY(), 300, [-7.5, 7.5], 2, label='AbsY')
        self.AxesLabels(2, 'Y [mm]', 'Entries')
        self.AxesTitle(2, 'Y projections')

    def UpdateRun(self, Run):
        self.data = Run

    def GShow(self):
        plt.show()

    def __del__(self):
        plt.close(self.figure)






        
# Test
if __name__ == '__main__':
    # TEST 1
    expr = "[0] + exp([1]*x)"
    fun = fit.GF1(expr)
    x = 1.
    param = (0., 2.)
    print(np.exp(2))
    print(fun(x, *param))
    print()

    # TEST 2
    #path = "/data2/XCF/gpd_data/GPD35/Ott24/Corrected/0Deg/911_0000809_data_0Deg_cn25325_tmc55.fits"
    path = "/data2/XCF/gpd_data/GPD35/Apr25/deg90/"
    Run0 = GRun(path)
    Canvas = GCanvas()

    # TEST 3
    par, cov, gaussian, chi2 = Run0.CutUnderPeak(2)
    Canvas.DataMaps(Run0)
    Canvas.AddFunctionPlot(1, Run0.GetPha(), gaussian, par, sigma=np.sqrt(np.diag(cov)), label='Peak Cut', chi2=chi2)

    ProjCanvas = GCanvas()
    ProjCanvas.Projections(data=Run0)
    spot_par, spot_cov, _, chi2 = Run0.FindGaussCenter()
    ProjCanvas.AddFunctionPlot(1, Run0.GetAbsX(), gaussian, spot_par[0], label='X gaussian', sigma=None, chi2=chi2[0])
    ProjCanvas.AddFunctionPlot(2, Run0.GetAbsY(), gaussian, spot_par[1], label='Y gaussian', sigma=None, chi2=chi2[1])

    QUCanvas = GCanvas()



    QUCanvas.GShow()

    # TEST 4
    #myGPD = gpd.GPD()
    #myGPD.ReadSpurious('/home/users/frassa/SpMaps/Smap_MoEn.fits')
    #Run0.MakeMaps(myGPD)
    #Run0.CutTheEdges()
    #Canvas2 = GCanvas()
    #Canvas2.DataMaps(Run0)
    #Canvas2.GShow()
