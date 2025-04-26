import numpy as np
import matplotlib.pyplot as plt 
import xarray as xr
import os
# print(os.path.abspath("."))
# units = (time, lev, lat, lon)

    #change python directory
os.chdir("/Users/jacealloway/Desktop/ Summer 2024/python:bash:GEOS-Chem")



def write_data():
        #loading / writing data with string formats
    dates = ['01', '02', '03', '04', '04', '05']    #list of dates
        #write len(dates) random 2x2 .txt arrays
    for date in dates:
        save_array = np.random.random((2, 2))
        np.savetxt("201901%s.yourmom"%date, save_array)

        #load the previously generated arrays, re-append them with all 1's, re-save them
    for date in dates:
        array = np.loadtxt("201901%s.yourmom"%date)
        array = np.ones((2,2))
        np.savetxt("201901%s.ones"%date, array)




def load_wind():
        #load files
    Met_U = np.load('20190101_Met_U.npy')
    Met_V = np.load('20190101_Met_V.npy')
        #isolate time
    Met_U = Met_U[0] 
    Met_V = Met_V[0]

    """
        print(Met_U.shape, Met_V.shape)
        (72, 91, 144)
        (72 levels,    180/2 + 1[equator] latitude,     360/2.5 longitude)
    """

        #isolate a specific level
    lat = np.linspace(0, 91, 91)
    lon = np.linspace(0, 144, 144)
    x, y = np.meshgrid(lat, lon)
    level = 5 #level out of 72

    plt.quiver(x, y, Met_U[level], Met_V[level], pivot = 'tail', headwidth = 10, headlength = 5, linewidth=0.5)
    plt.show()







"""

statemet = xr.open_mfdataset('/users/jk/20/jalloway/GC_v14.4.0/gc_2x25_geosfp_carbon_CO/OutputDir/GEOSChem.StateMet.20190101_*.nc4') #open statemet 

concentration = xr.open_mfdataset('/users/jk/20/jalloway/GC_v14.4.0/gc_2x25_geosfp_carbon_CO/OutputDir/GEOSChem.SpeciesConc.20190101_*.nc4') #open species  concentration        

        #isolate desired
co_data = concentration['SpeciesConcVV_CO']
press_data = statemet['Met_PMID']
U_data = statemet['Met_U']
V_data = statemet['Met_V']

        #.sel to extract values and index with tick nums
co_profile = co_data.sel(time="2019-01-01", lat=44.0, lon=-80)
press_profile = press_data.sel(time="2019-01-01", lat=44.0, lon=-80)
#co_profile = co_data.sel(time="2019-01-01", lat=0, lon=-20)
#press_profile = press_data.sel(time="2019-01-01", lat=0, lon=-20)

U_winds = U_data.values
V_winds = V_data.values
print(U_winds)
print(V_winds)

np.save("20190101_Met_U", U_winds)
np.save("20190101_Met_V", V_winds)




        #plot the profile
def main():
    fig, ax = plt.subplots(1,1)
    ax.plot(co_profile, press_profile)
    ax.invert_yaxis()
    ax.set_ylim(1000, 100)
    ax.set_xlabel("CO Concentration")
    ax.set_ylabel("Pressure Level")
    plt.savefig("Profile_Plot_2")
    plt.close()

    co_section = co_data.sel(lev = 6.918e-03, method='nearest')
    co_section.plot()
    plt.savefig("Vertical_2")

    
    
    
"""




    #loading A3dyn MERRA-2 Met Fields; re-write them only changing met_U, met_V vars
    #directory  /data/high_res/CTM/0.25x0.3125/GEOS_FP/2018/07    for July/2018



#  GEOSFP.YYYYMMDD.A3dyn.2x25.nc
#  at $METDIR/$YYYY/$MM/$MET.$YYYY$MM$DD.A3dyn.$RES.$NC
#  GEOSFP.YYYYMMDD.uv-wind.regrid.2x25.nc
#  at /data/high_res/jalloway/regrid_uv/output
#
#  hence relabel input directory as 
#  $A3DYNDIR/$MET.$YYYY$MM$DD.uv-wind.regrid.$RES.$NC
#  with A3DYNDIR as /data/high_res/jalloway/regrid_uv/output


