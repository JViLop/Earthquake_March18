
import pandas as pd
from obspy.core.utcdatetime import UTCDateTime
from datetime import datetime
from utils import utils
import os


main_dir = os.path.abspath(os.getcwd()) 
data_folder_dir =  os.path.join(main_dir,"data")
utils_dir = os.path.join(main_dir,"utils")




### Importing SAC files ###


data_files_list = os.listdir(data_folder_dir)
stations_name = list({file[3:7] for file in data_files_list})

### Defining event origin time ###


t0 = '2023-03-18 17:12:53.2'
event_origin = datetime.strptime(t0, "%Y-%m-%d %H:%M:%S.%f")
t0 = event_origin.isoformat()                     #ISO format
t0 = UTCDateTime(t0)                              # method to convert date into compatible trace fotmat



### TO DO:
    ### Include inputs: (path to folder (type of graph), x_lim,linear,logarithmic)



################################################################################
"""  P-Wave and S-Wave arrival times   """


cols = ['sta','net','dist','azi','phase','time','res','n','wt','stas'] 
df = pd.read_fwf('times.txt',header=None,
                 names=cols)
df.index = df['sta']
rows = df.loc[df['sta'].isin(stations_name)][['sta','dist','azi','phase','time']]

stations_data = dict()


for station in stations_name:
        stations_data[station]= dict()
        stations_data[station]['distance'] = float(rows.loc[station,'dist'][0])
        stations_data[station]['azimuth'] = float(rows.loc[station,'azi'][0])
        stations_data[station]['P-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+
                     rows.loc[station,'time'][0],"%Y-%m-%d %H:%M:%S.%f").isoformat())
        stations_data[station]['S-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+
                     rows.loc[station,'time'][1],"%Y-%m-%d %H:%M:%S.%f").isoformat())


###############################################################################

""" 
RAW SPECTRUM 
For each station, raw spectrum is obtained; processing includes only detrending
"""

utils.raw_spectrum_plots(main_dir,data_files_list,10,'linear',t0)

"""    NOISE SPECTRUM   (before P-wave arrival) """

utils.noise_spectrum_plots(main_dir,data_files_list,10,'linear',
                           stations_data,t0,0,0)


"""    P-WAVE SPECTRUM    """

utils.p_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
                            stations_data,t0)


"""    S-WAVE SPECTRUM  """

utils.s_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
                            stations_data,t0)


"""   CODA-WAVE SPECTRUM    """

utils.coda_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
                               stations_data,t0)




""" Effective duration by Husid (1969) """


utils.intensity_plots(main_dir,data_files_list,stations_data,t0)

