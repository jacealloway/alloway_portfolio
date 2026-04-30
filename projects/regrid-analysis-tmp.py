import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation, FFMpegWriter
import xarray as xr

""" 
20180701 - 20180801

DIRECTORIES:

2x25 initial
/users/jk/20/jalloway/GC_v14.4.0/gc_2x25_geosfp_carbon_CO/OutputDir/ *

2x25 regrid
/users/jk/20/jalloway/GC_v14.4.0/gc_2x25_geosfp_carbon_CO_regrid/OutputDir/ *

statemet-format
GEOSChem.StateMet.201807*.nc4

speccon-format
GEOSChem.SpeciesConc.201807*.nc4
"""

    #functions
def set_axis(ax):
    ax.cla()
    ax1.set_title("Initial")
    ax2.set_title("Regrid")
    ax.set_ylabel("Latitude")
    ax.set_xlabel("Longitude")







    #open the initial run files
statemet_init = xr.open_mfdataset('/users/jk/20/jalloway/GC_v14.4.0/gc_2x25_geosfp_carbon_CO/OutputDir/GEOSChem.StateMet.201807*.nc4')
speccon_init = xr.open_mfdataset('/users/jk/20/jalloway/GC_v14.4.0/gc_2x25_geosfp_carbon_CO/OutputDir/GEOSChem.SpeciesConc.201807*.nc4')
co_data_init = speccon_init['SpeciesConcVV_CO'] #isolate CO concentrations
press_data_init = statemet_init['Met_PMID']

    #open the regridded run files
statemet_regrid = xr.open_mfdataset('/users/jk/20/jalloway/GC_v14.4.0/gc_2x25_geosfp_carbon_CO_regrid/OutputDir/GEOSChem.StateMet.201807*.nc4')
speccon_regrid = xr.open_mfdataset('/users/jk/20/jalloway/GC_v14.4.0/gc_2x25_geosfp_carbon_CO_regrid/OutputDir/GEOSChem.SpeciesConc.201807*.nc4')
co_data_regrid = speccon_regrid['SpeciesConcVV_CO'] #isolate CO concentrations
press_data_regrid = statemet_regrid['Met_PMID']

    #files written once every hour at this resolution; same restart file used 
    #
    #dims: 	time = UNLIMITED ;  24x31 currently         in format ' 2018-07-01T01:00:00.000000000 '
	#       lev = 72 ;
	#       ilev = 73 ;
	#       lat = 91 ;
	#       lon = 144 ;
	#       nb = 2 ;
    #
    # STATEMET:
    #winds: Met_U, Met_V
    #pressure: Met_PMID
    #
    # SPECCON:
    #carbon-concentration: SpeciesConcVV_CO
    #
    # to get values use .values()
    # to isolate cols use .sel(time, lev, lat, lon) can use method='nearest' to round to closest actual value

hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
months = ['07']
time_array = []
for i in months:
    for j in days:
        for k in hours:
            time_array.append('2018-'+i+'-'+j+'T'+k+':00:00.000000000')

time_array.append('2018-08-01T01:00:00.000000000')



    #animation(s)
fig, (ax1, ax2) = plt.subplots(1,2)
def animate(frame):
    set_axis(ax1)
    set_axis(ax2)
    co_data_init.sel(time_array[frame])


    #check the vertical profiles of a few grid areas 


lat = (44.0, 0, -60, 8)
lon = (-80, -20, 0, 40)
fig, ((ax1,  ax2), (ax3, ax4))= plt.subplots(2,2, figsize=(20,20))

def set_axis():
    ax1.cla()
    ax2.cla()
    ax3.cla()
    ax4.cla()
    ax1.set_title("CO Profile at Lat = ",lat[0], "Lon = ", lon[0])
    ax2.set_title("CO Profile at Lat = ",lat[1], "Lon = ", lon[1])
    ax3.set_title("CO Profile at Lat = ",lat[2], "Lon = ", lon[2])
    ax4.set_title("CO Profile at Lat = ",lat[3], "Lon = ", lon[3])
    ax1.set_xlabel("CO Concentration (mol/mol dry)")
    ax1.set_ylabel("Pressure Level (hPa)")
    ax2.set_xlabel("CO Concentration (mol/mol dry)")
    ax2.set_ylabel("Pressure Level (hPa)")
    ax3.set_xlabel("CO Concentration (mol/mol dry)")
    ax3.set_ylabel("Pressure Level (hPa)")
    ax4.set_xlabel("CO Concentration (mol/mol dry)")
    ax4.set_ylabel("Pressure Level (hPa)")


def prof_ani(frame):
    set_axis()
    k = frame%len(time_array)
    co_profile_init_1 = co_data_init.sel(time = time_array[k], lat = lat[0], lon = lon[0])
    press_profile_init_1 = press_data_init.sel(time = time_array[k], lat = lat[0], lon = lon[0])
    co_profile_regrid_1 = co_data_regrid.sel(time = time_array[k], lat = lat[0], lon = lon[0])
    press_profile_regrid_1 = press_data_regrid.setl(time = time_array[k], lat = lat[0], lon = lon[0])

    co_profile_init_2 = co_data_init.sel(time = time_array[k], lat = lat[1], lon = lon[1])
    co_profile_regrid_2 = co_data_init.sel(time = time_array[k], lat = lat[1], lon = lon[1])

    co_profile_init_3 = co_data_init.sel(time = time_array[k], lat = lat[2], lon = lon[2])
    co_profile_regrid_3 = co_data_init.sel(time = time_array[k], lat = lat[2], lon = lon[2])

    co_profile_init_4 = co_data_init.sel(time = time_array[k], lat = lat[3], lon = lon[3])
    co_profile_regrid_4 = co_data_init.sel(time = time_array[k], lat = lat[3], lon = lon[3])

    ax1.plot(co_profile_init_1, label="Initial", color='blue')
    ax1.plot(co_profile_regrid_1, label="Regrid", color='magenta')
    ax1.legend()

    ax2.plot(co_profile_init_2, label="Initial", color='blue')
    ax2.plot(co_profile_regrid_2, label="Regrid", color='magenta')
    ax2.legend()

    ax3.plot(co_profile_init_3, label="Initial", color='blue')
    ax3.plot(co_profile_regrid_3, label="Regrid", color='magenta')
    ax3.legend()

    ax4.plot(co_profile_init_4, label="Initial", color='blue')
    ax4.plot(co_profile_regrid_4, label="Regrid", color='magenta')
    ax4.legend()


ani = FuncAnimation(fig, prof_ani, frames = 2000, interval = 1)
wr=FFMpegWriter(fps=30)
ani.save('/users/jk/20/jalloway/python_repo/co-profiles.mp4', writer=wr)
plt.close()






if isolated_profiles:
    t1 = time()
    print("Plotting isolated profiles...")

    num_plots = (1, 2, 3)
    timestamps = ('2018-07-11T17:00:00.000000000', '2018-07-16T12:00:00.000000000', '2018-07-14T21:00:00.000000000')
    lats = (0, -62, -36 )
    lons = (20, 120, 102.5)
    xlims = ((0, 0.5e-6), (0.5e-7, 0.75e-7), (0.5e-7, 1e-7))
    ylims = ((400, 800), (400, 800), (500, 850))
    titles=('co_prof_20180711-1700.png', 'co_prof_20180716-1200.png', 'co_prof_20180714-2100.png')

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    for i in range(len(num_plots)):
        co_profile_init = co_data_init.sel(time = timestamps[i], lat = lats[i], lon = lons[i], method='nearest')
        co_profile_regrid = co_data_regrid.sel(time = timestamps[i], lat = lats[i], lon = lons[i], method='nearest')
        press_profile_init = press_data_init.sel(time = timestamps[i], lat = lats[i], lon = lons[i], method='nearest')
        press_profile_regrid = press_data_regrid.sel(time = timestamps[i], lat = lats[i], lon = lons[i], method='nearest')
        
        ax.plot(co_profile_init, press_profile_init, label='Initial Run', color='magenta')
        ax.plot(co_profile_regrid, press_profile_regrid, label='Regridded Run', color='blue')
        ax.set_xlim(xlims[i])
        ax.set_xlabel('Carbon Concentration (mol/mol dry)', fontsize=12)
        ax.set_ylim(ylims[i])
        ax.set_ylabel('Pressure Level (hPa)', fontsize=12)
        ax.invert_yaxis()
        plt.legend(loc='upper right', fontsize=12)
        plt.savefig('/users/jk/20/jalloway/python_repo/'+titles[i])



    t2 = time()
    print("Profiles plotted in", t2 - t1, "seconds.")