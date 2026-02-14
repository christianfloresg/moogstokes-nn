"""
From a private communication with Christian. I modified it a bit so it will
take iSHELL FITS file and the BT-Settl templates. Wavelength units are all
converted to Angstroms. The BT-Settl spectra are also in air wavelengths by
default. If a template spectrum is loaded, the wavelengths will be shifted so 
the saved *.nspec spectrum is in terms of vacuum wavelengths.

This is a nice interactive continuum normalization routine that Christian found
in Python4Astronomers:
ftp://ftp.ster.kuleuven.be/dist/pierre/Mike/IvSPythonDoc/plotting/specnorm.html

Usage:
1-Click with the left button on the continuum part of the spectrum,
2-when enough points were selected, press the "enter" key, to fit
a polinomial
3-if you are happy with the polinomial press the "n" key to normalize
4-To write the normalize spectrum into a data file, press "w"
---
5- Click with the right button to un-select points it the spectrum
6- Press "r" key at any time to reset to the original spectrum
"""

# import pylab as plt
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splrep,splev
import sys
import os
from scipy.signal import savgol_filter,spline_filter
from scipy import interpolate
from fnmatch import fnmatch

from spectra import *


BAND_W_LIMS = {'J':[12600, 13700], 'K':[20800, 24000], 'M':[45100, 52300]}
continuum = []


def onclick(event):
    # when none of the toolbar buttons is activated and the user clicks in the
    # plot somewhere, compute the median value of the spectrum in a 10angstrom
    # window around the x-coordinate of the clicked point. The y coordinate
    # of the clicked point is not important. Make sure the continuum points
    # `feel` it when it gets clicked, set the `feel-radius` (picker) to 5 points
    toolbar = plt.get_current_fig_manager().toolbar
    if event.button==1 and toolbar.mode=='':
        count=0
        #window = ((event.xdata-2)<=wave) & (wave<=(event.xdata+2))
        # window = ((event.xdata-5.e-4)<=wave) & (wave<=(event.xdata+5.e-4))
        #y = np.median(flux[window])
        y = event.ydata
        plt.plot(event.xdata,y,'gs',ms=5,picker=2,label='cont_pnt')
    plt.draw()


def onpick(event):
    # when the user clicks right on a continuum point, remove it
    if event.mouseevent.button==3:
        if hasattr(event.artist,'get_label') and event.artist.get_label()=='cont_pnt':
            event.artist.remove()

def ontype(event):
    # when the user hits enter:
    # 1. Cycle through the artists in the current axes. If it is a continuum
    #    point, remember its coordinates. If it is the fitted continuum from the
    #    previous step, remove it
    # 2. sort the continuum-point-array according to the x-values
    # 3. fit a spline and evaluate it in the wavelength points
    # 4. plot the continuum
    if event.key=='enter':
        cont_pnt_coord = []
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and artist.get_label()=='cont_pnt':
                cont_pnt_coord.append(artist.get_data())
            elif hasattr(artist,'get_label') and artist.get_label()=='continuum':
                artist.remove()
        cont_pnt_coord = np.array(cont_pnt_coord)[...,0]
        sort_array = np.argsort(cont_pnt_coord[:,0])
        x,y = cont_pnt_coord[sort_array].T
        spline = splrep(x,y,k=1)
        continuum = splev(wave,spline)
        plt.plot(wave,continuum,'r-',lw=2,label='continuum')
    # when the user hits 'n' and a spline-continuum is fitted, normalise the
    if event.key=='n':
        continuum = None
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and artist.get_label()=='continuum':
                continuum = artist.get_data()[1]
                break
        if continuum is not None:
            plt.cla()
            plt.plot(wave,flux/continuum,'k-',label='normalised')
            plt.plot(wave,uncert/continuum,'r-',label='uncertainty')
            ymed = np.nanmedian(flux/continuum)
            plt.ylim(ymed/2, ymed*1.5)
            plt.plot(wave, np.ones_like(wave), c='cyan', alpha=0.5)
  
    # when the user hits 'r': clear the axes and plot the original spectrum
    elif event.key=='r':
        plt.cla()
        plt.plot(wave,flux,'k-')
        ymed = np.nanmedian(flux)
        plt.ylim(ymed/2, ymed*1.5)

    elif event.key == 'w':
        normalized_flux = []
        normalized_err = []
        for artist in plt.gca().get_children():
            if hasattr(artist,'get_label') and artist.get_label()=='normalised':
                data = np.array(artist.get_data())
                normalized_flux = data[1]
                # np.savetxt(os.path.splitext(filename)[0]+'.nspec',data.T)
                print('flux Saved to file')
                # break

            elif hasattr(artist,'get_label') and artist.get_label()=='uncertainty':
                data2 = np.array(artist.get_data())
                normalized_err = data2[1]
                # np.savetxt(os.path.splitext(filename)[0]+'.nspec',data.T)
                print('error Saved to file')
                break

        save = open(os.path.splitext(filename)[0]+'.nspec', "w")
        w_length = len(wave)
        for ii in range(w_length):
            save.write('{:15} {:>20} {:>20}\n'.format(wave[ii]*1e4, normalized_flux[ii], normalized_err[ii]))
        save.close()
        sys.exit()
    plt.draw()


if __name__ == "__main__":
    # Get the filename of the spectrum from the command line, and plot it
    filename = sys.argv[1]
    data = ProplydData(filename)
    wave, flux, uncert = data.x, data.y, data.yerr

    # x = np.isnan(flux)
    # flux[x]=np.nanmedian(flux)
    # smoothed_flux=savgol_filter(flux, window_length=27, polyorder=2)
    spectrum, = plt.plot(wave,flux,'k-',label='spectrum',linewidth=0.7)
    ymed = np.nanmedian(flux)
    plt.ylim(ymed/2, ymed*1.5)
    # spectrum, = plt.plot(wave,smoothed_flux,'r-',label='smoothed',linewidth=0.7)
    

    plt.title(filename)
    # Connect the different functions to the different events
    plt.gcf().canvas.mpl_connect('button_press_event',onclick)
    plt.gcf().canvas.mpl_connect('pick_event',onpick)
    plt.gcf().canvas.mpl_connect('key_press_event',ontype)
    plt.show() # show the window
    # plt.ion()
