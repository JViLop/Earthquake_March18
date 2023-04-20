
import pandas as pd
from obspy.core.utcdatetime import UTCDateTime
import matplotlib.pylab as plt
import numpy as np
from datetime import datetime
from utils import utils
import re
import os




### Setting up general information ###


path_event_text_file = "D:/Proyectos/Earthquake_20230318/Earthquake_March18/igepn2023fkei.txt"

def line_index(path,string):
    with open(path, 'r') as fp:
        for l_no, line in enumerate(fp):
            # search string
            if string in line:
                c =l_no
                break
                # don't look for next lines
    return c
    
def string_finder(path,string):
    with open(path, 'r') as fp:
        for l_no, line in enumerate(fp):
            # search string
            if string in line: 
                c = line
                break
                # don't look for next lines
    return c
        
event_id = string_finder(path_event_text_file,'Public ID')
event_id_regex = re.compile(r'(\w{5}\d{4}\w{4})')
event_id = re.findall(event_id_regex,event_id)[0]

date = string_finder(path_event_text_file,'Date') 
date_regex = re.compile(r'(\d{4}-\d{2}-\d{2})')
date = re.findall(date_regex,date)[0]

time = string_finder(path_event_text_file,'Time') 
time_regex = re.compile(r'(\d{2}:\d{2}:\d*\.?\d*)')
time = re.findall(time_regex,time)[0]
   
latitude = string_finder(path_event_text_file,'Latitude') 
latitude_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
latitude = re.findall(latitude_regex,latitude)[0]
   
longitude = string_finder(path_event_text_file,'Longitude') 
longitude_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
longitude = re.findall(longitude_regex,longitude)[0]


depth = string_finder(path_event_text_file,'Depth') 
depth_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
depth = re.findall(depth_regex,depth)[0]



lineStationPhase = line_index(path_event_text_file,'Phase arrivals:')  
lineStationMagn = line_index(path_event_text_file,'Station magnitudes:')

       
main_dir = os.path.abspath(os.getcwd()) 
data_folder_dir =  os.path.join(main_dir,"data")
plots_folder_dir =  os.path.join(main_dir,"plots")
utils_dir = os.path.join(main_dir,"utils")



data_el_oro_folder_dir=  os.path.join(main_dir,"data_ElOro")
data_el_oro_files_list = os.listdir(data_el_oro_folder_dir)
### Importing SAC files ###


data_files_list = os.listdir(data_folder_dir)
stations_name = list({file[3:7] for file in data_files_list})
stations_el_oro_name = list({file[4:8] for file in data_el_oro_files_list})


### Defining event origin time and origin depth###


t0 = '2023-03-18 17:12:53.2'
event_origin = datetime.strptime(t0, "%Y-%m-%d %H:%M:%S.%f")
t0 = event_origin.isoformat()                     #ISO format
t0 = UTCDateTime(t0)                              # method to convert date into compatible trace fotmat
depth = 60

fc = 24



################################################################################
"""  P-Wave and S-Wave arrival times   """


cols = ['sta','net','dist','azi','phase','time','res','n','wt','stas']
df = pd.read_fwf('times.txt',header=None,
                 names=cols)
df.index = df['sta']
rows = df.loc[df['sta'].isin(stations_name)][['sta','dist','azi','phase','time']]

stations_data = dict()

cols1 =["NET.STATION.LOCID.CMPNM", "MaxAcc", "TimeOfMaxAcc", "DIST", "AZ", "BAZ",
        "PeakVecACC", "EstimatedDuration."]
df1 = pd.read_csv("summary_ACC.txt",names=cols1,sep=' ',skiprows=1)
df1['station'] = df1['NET.STATION.LOCID.CMPNM'].str[3:7]
df1['CHN'] = df1['NET.STATION.LOCID.CMPNM'].str[-3:]
rows1 = df1.loc[df1['station'].isin(stations_name)][["station","CHN","MaxAcc", "TimeOfMaxAcc", "DIST"]]
rows1['stationCHN'] = rows1['station']  + rows1['CHN']  
rows1 = rows1.set_index('stationCHN')  
rows1 = rows1.loc[rows1['CHN'].isin(['HNE','HNN','HNZ','BLE','BLN','BLZ'])][["station","CHN","MaxAcc", "TimeOfMaxAcc", "DIST"]]
rows1['CHN'] = rows1['CHN'].str[-1:]
rows1 = rows1.set_index(rows1['station']  + rows1['CHN'])  



for station in stations_name:
        stations_data[station]= dict()
        stations_data[station]['azimuth'] = float(rows.loc[station,'azi'][0])
        stations_data[station]['P-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+
                     rows.loc[station,'time'][0],"%Y-%m-%d %H:%M:%S.%f").isoformat())
        stations_data[station]['S-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+
                     rows.loc[station,'time'][1],"%Y-%m-%d %H:%M:%S.%f").isoformat())
        stations_data[station]['maxAccZ'] = float(rows1.loc[station+'Z','MaxAcc'])
        stations_data[station]['maxAccN'] = float(rows1.loc[station+'N','MaxAcc'])
        stations_data[station]['maxAccE'] = float(rows1.loc[station+'E','MaxAcc'])
        stations_data[station]['TimeOfmaxAccZ'] = UTCDateTime(datetime.strptime('2023-03-18 '+
                    rows1.loc[station+'Z','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
        stations_data[station]['TimeOfmaxAccN'] =  UTCDateTime(datetime.strptime('2023-03-18 '+
                    rows1.loc[station+'N','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
        stations_data[station]['TimeOfmaxAccE'] =  UTCDateTime(datetime.strptime('2023-03-18 '+
                    rows1.loc[station+'E','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
        stations_data[station]['epicentralDist'] = float(rows1.loc[station+'E','DIST'])
        stations_data[station]['hypocentralDist'] = round(np.sqrt(float(rows1.loc[station+'E','DIST'])**2 + depth**2),2)

###############################################################################

# """ 
# RAW SPECTRUM 
# For each station, raw spectrum is obtained; processing includes only detrending
# """

# utils.raw_spectrum_plots(main_dir,data_files_list,10,'linear',t0)

# """    NOISE SPECTRUM   (before P-wave arrival) """

# utils.noise_spectrum_plots(main_dir,data_files_list,10,'linear',
#                             stations_data,t0,0,0)


# """    P-WAVE SPECTRUM    """

# utils.p_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
#                             stations_data,t0)


# """    S-WAVE SPECTRUM  """

# utils.s_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
#                             stations_data,t0)


# """   CODA-WAVE SPECTRUM    """

# utils.coda_wave_spectrum_plots(main_dir,data_files_list,10,'linear',
#                                 stations_data,t0)

# #inicio 3:42pm


""" Effective duration by Husid (1969) """


# utils.intensity_plots(main_dir,data_el_oro_files_list,stations_data,t0)


"""  CODA-Q ANALYSIS"""

""" Linear Regression"""

# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,0.75,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,1.5,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,3,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,6,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,12,factor_e=6)
# utils.coda_analysis_spectrum_plots(main_dir,data_files_list,stations_data,t0,24,factor_e=6)


# csv_folder_dir =os.path.join(main_dir,"csv_files")
# csv_folder_dir =os.path.join(csv_folder_dir,"Coda_Q")
# csv_files_list = os.listdir(csv_folder_dir)
# Qmodel_dir = os.path.join(plots_folder_dir,"Coda_Q")

# freqs = [0.75*2**(i) for i in range(6)]



# fig,axes = plt.subplots(15,3,figsize=(12,16))
# for j,station in enumerate(stations_name):
#     for k,comp in enumerate(['E','N','Z']):
#         Q=[]
#         for i in range(6):
#             path = os.path.join(csv_folder_dir,csv_files_list[i])
#             df = pd.read_csv(os.path.join(csv_folder_dir,csv_files_list[i]),index_col=0)                         
#             Qc=df.loc[station][comp]
#             Q.append(Qc)   
#         axes[j][k].plot(freqs,Q,'-o',linewidth=0.4,markersize=1.5)
#         axes[j][k].annotate(station,xy=(freqs[1],Q[1]),xytext=(freqs[1],Q[1]),ha='center',va='top',fontsize=1)   
#         axes[j][k].set_title(station + ' ' + comp,fontdict={'fontsize':8,'color':'blue'})


# fig.suptitle('Event: igepn2023fkei Time: {} \n  Q factor as functio of f'.format(t0))
# fig.supylabel(r'$Q(f)$')
# fig.supxlabel(r'f')
# fig.tight_layout(pad=1.25)
# fig.savefig(Qmodel_dir+'/Qmodel.jpg',dpi=600)










