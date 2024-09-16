
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr




def plot_swot_basemap(ax, xlims, ylims, fine_contours=False, swath=True):

    # Add bathy etc to the plot
    basedir = '../../MNF_SWOT/GIS'
    swathpoly = gpd.read_file("{}/SWOT_calval_Browse_overpass_swath.shp".format(basedir))
    swathline = gpd.read_file("{}/SWOT_calval_Browse_overpass_nadir.shp".format(basedir))
    
    # Load some bathy data
    dsall = xr.open_dataset('~/data/Bathymetry/GA_WEL_NWS_250m_DEM.nc')
    dsZ = dsall.assign_coords(nx=dsall.X,ny=dsall.Y).sel(nx=slice(xlims[0],xlims[1]), ny=slice(ylims[0], ylims[1]))
    
    c= plt.contour(dsZ['X'],dsZ['Y'],-dsZ['topo'],[100,200,300,400, 500],colors='k',linewidths=0.5)
    if fine_contours:
        c= plt.contour(dsZ['X'],dsZ['Y'],-dsZ['topo'],np.arange(100,500,10),colors='0.5',linewidths=0.2)

    p1=plt.plot(123.16238333,-14.23543333,'rd') # BRB200 mooring=
    p4=plt.plot(123.03041737634493, -14.230653066337094,'bo', markeredgecolor='k') # S245
    p5=plt.plot(122.8370658081835, -14.13718816307405,'bo', markeredgecolor='k') # W310
    p6=plt.plot(123.02928797854348, -14.052341197573492,'bo', markeredgecolor='k') # N280

    if swath:
        p2=swathpoly.plot(ax=ax, facecolor='#859101', alpha=0.25, zorder=1e6)


    plt.xlim(xlims)
    plt.ylim(ylims)
    ax.set_aspect('equal')  

    del swathpoly
    del swathline
    del dsZ
    
    return c

### s3 file handling
def open_file_nocache(fname, myfs):
    """
    Load a netcdf file directly from an S3 bucket
    """
    fileobj = myfs.open(fname)
    return xr.open_dataset(fileobj)

def open_mfile_nocache(fnames, myfs):
    """
    Load a netcdf file directly from an S3 bucket
    """
    fileobjs = [myfs.open(fname) for fname in fnames]
    return xr.open_mfdataset(fileobjs)

### Signal processing
from scipy import signal

def filt(ytmp, cutoff_dt, dt, btype='low', order=8, ftype='sos', axis=-1):
    """
    Butterworth filter the time series

    Inputs:
        cutoff_dt - cuttoff period [seconds]
        btype - 'low' or 'high' or 'band'
    """
    if not btype == 'band':
        Wn = dt/cutoff_dt
    else:
        Wn = [dt/co for co in cutoff_dt]
    
    if ftype=='sos':
        sos = signal.butter(order, Wn, btype, analog=False, output='sos')
        return signal.sosfiltfilt(sos, ytmp, axis=axis)
    else:
        (b, a) = signal.butter(order, Wn, btype=btype, analog=0, output='ba')
        return signal.filtfilt(b, a, ytmp, axis=axis)


def filt_decompose(xraw, dt, b1=34*3600, b2=4*3600):
    x1 = filt(xraw,b1, dt, btype='low')
    x2 = filt(xraw, [b1,b2], dt, btype='band')
    x3 = filt(xraw, b2, dt, btype='high')
    
    xin = np.vstack([x1,x2,x3]).T
    
    return xin
