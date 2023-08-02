"""
Utility functions to load Ponte and Klein, 2015 data
"""

import xarray as xr
import numpy as np
from glob import glob
from scipy.interpolate import interp1d


# Create nicer dimensions
def load_pk2015(ncfile,datavar='ssh'):
    ds = xr.open_dataset(ncfile)
    dx = 4.0
    if len(ds['{}_lof'.format(datavar)].dims)==4:
        tdim, zdim, ydim, xdim = ds['{}_lof'.format(datavar)].dims
    else:
        tdim, ydim, xdim = ds['{}_lof'.format(datavar)].dims
    x = ds[xdim].values*dx
    y = ds[ydim].values*dx

    t0 = np.datetime64('1950-01-01 00:00:00')
    tdays = (ds.time_centered.values - t0).astype(float)*1e-9/86400.
    print(ncfile, tdays[0], tdays[-1],ds.time_centered.values[0])
    return xr.Dataset(
        {'{}_lof'.format(datavar):
            xr.DataArray(ds['{}_lof'.format(datavar)].squeeze(), coords={'x':x,'y':y, 'time':tdays}, dims=('time','y','x')),
         '{}_cos'.format(datavar):
            xr.DataArray(ds['{}_cos'.format(datavar)].squeeze(), coords={'x':x,'y':y, 'time':tdays}, dims=('time','y','x')),
         '{}_sin'.format(datavar):
            xr.DataArray(ds['{}_sin'.format(datavar)].squeeze(), coords={'x':x,'y':y, 'time':tdays}, dims=('time','y','x')),
        })

def load_scenario(scenario, datadir):
        ds_v = xr.concat([load_pk2015(ff, datavar='v_xy') for ff in sorted(glob('{}/{}-t*.nc'.format(datadir, scenario)))], dim='time')
        ds_u = xr.concat([load_pk2015(ff, datavar='u_xy') for ff in sorted(glob('{}/{}-t*.nc'.format(datadir, scenario)))], dim='time')
        ds_ssh = xr.concat([load_pk2015(ff, datavar='ssh') for ff in sorted(glob('{}/{}-t*.nc'.format(datadir, scenario)))], dim='time')
        ds = xr.merge([ds_v, ds_u, ds_ssh])
        ds['time'] = np.arange(ds.time[0], ds.time[0]+ds.time.shape[0]*2, 2)

        return ds
    
def calc_raw(ds, xpt, ypt, dtout, varname):
    omega = 2*np.pi*2. # 2 cpd

    tfast = np.arange(ds.time.values[0], ds.time.values[-1], dtout)

    F = interp1d(ds.time.values, ds['{}_lof'.format(varname)].isel(x=xpt, y=ypt), kind=2)
    vlow = F(tfast)
    F = interp1d(ds.time.values, ds['{}_cos'.format(varname)].isel(x=xpt, y=ypt), kind=2)
    vcos = F(tfast)
    F = interp1d(ds.time.values, ds['{}_sin'.format(varname)].isel(x=xpt, y=ypt), kind=2)
    vsin = F(tfast)

    vraw = vlow + vcos*np.cos(omega*tfast) + vsin*np.sin(omega*tfast)
    
    return tfast, vraw, vlow

def subset_data(ds, yslice = slice(35,680, 2), xslice = slice(None, None, 10)):
    # Remove the mean from the cos and sin to leave the incoherent signal
    # ...also subset the domain for faster computation
        

    ds_nonstat = ds.isel(time=slice(10,150), y=yslice, x=xslice)
    ds_nonstat['ssh_cos'] = ds['ssh_cos'] - ds['ssh_cos'].mean(axis=0)
    ds_nonstat['ssh_sin'] = ds['ssh_sin'] - ds['ssh_sin'].mean(axis=0)

    ds_nonstat['ssh_cos_stat'] = ds['ssh_cos'].mean(axis=0)
    ds_nonstat['ssh_sin_stat'] = ds['ssh_sin'].mean(axis=0)

    return ds_nonstat