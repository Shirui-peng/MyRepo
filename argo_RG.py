## Import external packages and modules
import numpy as np
from scipy.interpolate import interp1d
from scipy import arange, exp
import xarray as xr
import gsw as gsw
import great_circle_calculator.great_circle_calculator as gcc
from scipy.interpolate import RegularGridInterpolator as RGI
from datetime import datetime,timedelta,date,timezone
import itertools
import functools
import utils

src,stn = 'Nias','H08'
name = src+'_'+stn
resolution = 'coarsen'

# Define path end points
if stn == 'H01':
    p10,p20 = (114.14, -34.88),(96.95, 1.12)
elif stn == 'H08':
    p10,p20 = (72.49, -7.65),(96.92, 1.62)
elif stn == 'DGAR':
    p10,p20 = (72.45, -7.41),(96.96, 1.63)
d0 = gcc.distance_between_points(p10, p20, unit='meters')
crs1,crs2 = gcc.bearing_at_p1(p10, p20),gcc.bearing_at_p1(p20, p10)

# output timestep, and output choices 
# for vertical anomalies, weighted anomalies
dtself,vert,wdtau,refself = 1,1,0,0

ds = xr.open_dataset(f'data/knl/nias_kernels/coarsen/KTs_'+name+'_'+resolution+'.nc')
fs = ds.f
idx_topo = np.where(ds.SEMkernels_T[0].values==0)
if resolution == 'coarsen':
    xk,zk = ds.x.values,ds.z.values
    xmaxk,xmink = max(xk),min(xk)
    nxk,nzk = len(xk),len(zk)
    dxk,dzk = np.ptp(xk)/(nxk-1),np.ptp(zk)/(nzk-1)
else:
    # start and end of path
    if stn != 'H01':
        xmink,xmaxk,dxk = -100e3,3000e3, 10e3 
    else:
        xmink,xmaxk,dxk = -100e3,4500e3, 5e3 

    zmink,zmaxk,dzk = -6500.,0.,50.

    # size of kernel array
    nxk,nzk = int((xmaxk-xmink)/dxk)+1,int((zmaxk-zmink)/dzk)+1

    # kernel coordinates
    #xk = np.linspace(xmink,xmaxk,nxk)
    #zk = np.linspace(zmink,zmaxk,nzk)
    #ds = ds.interp(x=xk, z=zk)

# kernel end points
p2 = gcc.point_given_start_and_bearing(p10, crs1, xmaxk, unit='meters')
p1 = gcc.point_given_start_and_bearing(p20, crs2, d0-xmink, unit='meters')
lonT1,lonT2 = p1[0],p2[0]
latT1,latT2 = p1[1],p2[1]

head = 'data/argo/'
t_argo = xr.open_dataset(f'{head}RG_ArgoClim_Temperature_2019.nc',decode_times=False)
#s_argo = xr.open_dataset(f'{head}RG_ArgoClim_Salinity_2019.nc',decode_times=False)
t_argo = t_argo.assign_coords(LOGITUDE=(((t_argo.LONGITUDE + 160) % 360) - 180))
#s_argo = s_argo.assign_coords(LOGITUDE=(((s_argo.LONGITUDE + 160) % 360) - 180))

#tref,sref = t_argo.ARGO_TEMPERATURE_MEAN,s_argo.ARGO_SALINITY_MEAN
lon_a,lat_a = t_argo.LONGITUDE,t_argo.LATITUDE
dlon_a = np.diff(lon_a).max()
dlat_a = np.diff(lat_a).max()
# reverse depth coordinate
za = t_argo.PRESSURE.values[::-1]

t_a = t_argo.ARGO_TEMPERATURE_ANOMALY

# Get (lon,lat) for the grid points along the path
fr = np.linspace(0,1,nxk)
gridT = np.array(list(map(functools.partial(utils.ip, p1=p1,p2=p2), fr)))
lat_k, lon_k = gridT[:,1],gridT[:,0]
i_k = np.where((lon_a<=max(p1[0],p2[0])+dlon_a) & (lon_a>=min(p1[0],p2[0])-dlon_a))[0]
j_k = np.where((lat_a<=max(p1[1],p2[1])+dlat_a) & (lat_a>=min(p1[1],p2[1])-dlat_a))[0]
t_a = t_a.isel(LONGITUDE=i_k,LATITUDE=j_k)

idx_lonk = np.where((lon_k>=lon_a[i_k].min().values) & (lon_k<=lon_a[i_k].max().values))[0]
idx_latk = np.where((lat_k>=lat_a[j_k].min().values) & (lat_k<=lat_a[j_k].max().values))[0]
idx_k = np.intersect1d(idx_lonk,idx_latk)

# all events
if dtself:
    dt64_events = t_argo.TIME.values

# kernel coordinates
Tk = np.full([len(dt64_events),nxk, nzk],np.nan)
if vert:
    dTtv = np.zeros([len(dt64_events),nzk])
    
if wdtau:
    dtauSEM = np.zeros([len(fs),len(dt64_events)])
    dtauMODE = np.zeros([len(fs),len(dt64_events)])
    
for e, dt64 in enumerate(dt64_events):
    if dtself:
        print(f'{e} of {len(dt64_events)}')
        Ta = t_a[e].values[::-1,:,:].T
    else:
        # find time stamps of ECCO data bracketing the event
        ts = (dt64 - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's')
        event = datetime.utcfromtimestamp(ts)
        dt_mid = utils.dt_midmonth(event.year,event.month)
        nt_a = 12*(event.year-2004)+event.month-1
        if event<dt_mid:
            nt_day1 = nt_a-1
            if event.month>1:
                day1 = utils.dt_midmonth(event.year,event.month-1)
            else:
                day1 = utils.dt_midmonth(event.year-1,12)
            day2 = dt_mid
        else:
            day1 = dt_mid
            nt_day1 = nt_a
            if event.month<12:
                day2 = utils.dt_midmonth(event.year,event.month+1)
            else:
                day2 = utils.dt_midmonth(event.year+1,1)

        nt_day2 = nt_day1+1

        print(day2)

        # mask missing data and reverse depth coordinate
        T1,T2 = T_a[nt_day1],T_a[nt_day2]

        # interpolate to the time of the event
        ddt = day2-day1
        w1 = (day2 - event)/ddt
        w2 = (event - day1)/ddt
        Ta = w1*T1 + w2*T2
        print(f'{w1} {w2}')
        
    # ECCO data array size
    nxa, nya, nza = Ta.shape
    # fill in the coastal and bottom points
    Ta = utils.filledges(Ta)
    
    # Get (lon,lat) for the grid points along the path
    # interpolate horizontally onto great circle path
    knots = (lon_a[i_k], lat_a[j_k])
    locsk = np.empty([nxk,2])
    locsk[:,0],locsk[:,1] = lon_k,lat_k
    Tkza = np.empty([nxk, nza])
    for k in range(nza):
        itpT = RGI(knots, Ta[:,:,k])
        Tkza[:,k] = itpT(locsk)
        
    # interpolate vertically onto kernel grid
    # fill in the coastal and bottom points
    Tkza = utils.fillbtm2(Tkza)
    knots = za
    for i in idx_k:
        itpT = interp1d(knots, Tkza[i,:])
        etpT = utils.extrap1d(itpT)
        Tk[e,i,:] = etpT(zk)
        Tk[e,i,zk<-2e3] = np.nan
    if refself != 1:
        # calculate in situ temperature
        dT = Tk[e]
        dT[idx_topo] = np.nan
        if vert:
            dTtv[e,:] = np.nanmean(dT,axis=0)
            dav = xr.DataArray(dTtv,[("t", dt64_events),("z", zk)],)
            dav.to_netcdf('results/argo/dTz_'+name+'_argo.nc')
        if wdtau:
            for j in range(len(fs)):
                dtauSEM[j,e] = np.nansum(ds.SEMkernels_T[j]*dT)*dxk*dzk
                dtauMODE[j,e] = np.nansum(ds.MODEkernels_T[j]*dT)*dxk*dzk
            dst = xr.Dataset(
                data_vars=dict(
                    SEMdtaus=(["f", "t"], dtauSEM),
                    MODEdtaus=(["f", "t"], dtauMODE), ),
                coords=dict(
                    f = fs.values,
                    t = dt64_events.values,),
                attrs=dict(description="Monthly travel time anomalies (s) starting from Jan 2004"),)
            dst.to_netcdf('results/argo/dtaus_'+name+'_argo_'+resolution+'KTs.nc')