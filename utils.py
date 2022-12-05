## Import external packages and modules
import numpy as np
from scipy.interpolate import interp1d
from scipy import arange, exp
import xarray as xr
import gsw as gsw
import great_circle_calculator.great_circle_calculator as gcc
from datetime import datetime,timedelta,date,timezone
import itertools
import functools

# fill in coastal and bottom points (ignores points at the edge of the array)
def filledges(a):
    # fill in all points that have a missing value with the average of the eight surrounding points (if there are any values there)
    nx, ny, nz = a.shape
    b = np.empty([nx, ny, nz])
    b[:,:,:] = a[:,:,:]
    for i,j,k in itertools.product(range(1,nx-1), range(1,ny-1), range(nz)):
        if (np.isnan(a[i,j,k]) and any(~np.isnan(a[i-1:i+2,j-1:j+2,k]).flatten())):
            b[i,j,k] = np.nanmean(a[i-1:i+2,j-1:j+2,k])
    # fill in bottom points
    for i,j,k in itertools.product(range(nx), range(ny), range(nz-1)):
        if (np.isnan(a[i,j,k]) and ~np.isnan(a[i,j,k+1])):
            b[i,j,k] = a[i,j,k+1]
    return b

# fill in bottom points along the path
def fillbtm(a,n):
    # fill in all points that have a missing value with the average of the eight surrounding points (if there are any values there)
    nx, nz = a.shape
    b = np.empty([nx, nz])
    b[:,:] = a[:,:]
    for i,k in itertools.product(range(nx), range(nz-1)):
        if (np.isnan(a[i,k]) and ~np.isnan(a[i,k+1])):
            b[i,k] = a[i,k+1]
            if n>0:
                b[i,max([0,k-n]):k] = a[i,k+1]
    return b

# fill in bottom points along the path
def fillbtm2(a):  
  # fill in all points that have a missing value with the average of the two surrounding points (if there are any values there)
    nx, nz = a.shape
    b = np.empty([nx, nz])
    b[:,:] = a[:,:]
    for i in range(nx):
        for k in range(nz-1):
            if (np.isnan(b[i,nz-k-2]) and ~np.isnan(b[i,nz-k-1])):
                b[i,nz-k-2] = b[i,nz-k-1]
    return b

def ip(frac,p1,p2):
    return gcc.intermediate_point(p1, p2, frac)

def cdif(x):
    return np.insert(x[2:]-x[:-2],[0,-1],[x[1]-x[0],x[-1]-x[-2]])

def grad(x,y):
    return cdif(y)/cdif(x)

def pT(T, S, lon, lat, p):
    SA = gsw.SA_from_SP(S, p, lon, lat)
    pT = gsw.pt0_from_t(SA, T, p)
    return pT
    
# 1d extrapolation
def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return np.array(list(map(pointwise, np.array(xs))))

    return ufunclike

def dt_midmonth(year,month):
    dt_start = datetime(year,month,1)
    if month<12:
        dt_end = datetime(year,month+1,1)
    else:
        dt_end = datetime(year+1,1,1)
    dt_mid = dt_start+(dt_end-dt_start)/2
    return dt_mid