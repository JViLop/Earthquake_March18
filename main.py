
import pandas as pd
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pylab as plt
from datetime import datetime
from utils import utils
import os


main_dir = os.path.abspath(os.getcwd()) 
data_folder_dir =  os.path.join(main_dir,"data")
plots_folder_dir =  os.path.join(main_dir,"plots")
utils_dir = os.path.join(main_dir,"utils")


### Importing SAC files ###


data_files_list = os.listdir(data_folder_dir)
stations_name = list({file[3:7] for file in data_files_list})

### Defining event origin time ###


t0 = '2023-03-18 17:12:53.2'
event_origin = datetime.strptime(t0, "%Y-%m-%d %H:%M:%S.%f")
t0 = event_origin.isoformat()                     #ISO format
t0 = UTCDateTime(t0)                              # method to convert date into compatible trace fotmat


fc = 24



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

# """ 
# RAW SPECTRUM 
# For each station, raw spectrum is obtained; processing includes only detrending
# """

# utils.raw_spectrum_plots(main_dir,data_files_list,10,'linear',t0)

# """    NOISE SPECTRUM   (before P-wave arrival) """

# utils.noise_spectrum_plots(main_dir,data_files_list,10,'linear',
#                            stations_data,t0,0,0)


# """    P-WAVE SPECTRUM    """

# utils.p_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
#                             stations_data,t0)


# """    S-WAVE SPECTRUM  """

# utils.s_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
#                             stations_data,t0)


# """   CODA-WAVE SPECTRUM    """

# utils.coda_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
#                                stations_data,t0)

# #inicio 3:42pm


# """ Effective duration by Husid (1969) """


# utils.intensity_plots(main_dir,data_files_list,stations_data,t0)

"""  CODA-Q ANALYSIS"""

""" Linear Regression"""

# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,0.75,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,1.5,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,3,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,6,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,12,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,24,factor_e=6)


csv_folder_dir =os.path.join(main_dir,"csv_files")
csv_folder_dir =os.path.join(csv_folder_dir,"Coda_Q")
csv_files_list = os.listdir(csv_folder_dir)
Qmodel_dir = os.path.join(plots_folder_dir,"Coda_Q")

freqs = [0.75*2**(i) for i in range(6)]



fig,axes = plt.subplots(15,3,figsize=(12,16))
for j,station in enumerate(stations_name):
    for k,comp in enumerate(['E','N','Z']):
        Q=[]
        for i in range(6):
            path = os.path.join(csv_folder_dir,csv_files_list[i])
            df = pd.read_csv(os.path.join(csv_folder_dir,csv_files_list[i]),index_col=0)                         
            Qc=df.loc[station][comp]
            Q.append(Qc)   
        axes[j][k].plot(freqs,Q,'-o',linewidth=0.4,markersize=1.5)
        axes[j][k].annotate(station,xy=(freqs[1],Q[1]),xytext=(freqs[1],Q[1]),ha='center',va='top',fontsize=1)   
        axes[j][k].set_title(station + ' ' + comp,fontdict={'fontsize':8,'color':'blue'})


fig.suptitle('Event: igepn2023fkei Time: {} \n  Q factor as functio of f'.format(t0))
fig.supylabel(r'$Q(f)$')
fig.supxlabel(r'f')
fig.tight_layout(pad=1.25)
fig.savefig(Qmodel_dir+'/Qmodel.jpg',dpi=600)










