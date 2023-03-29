import numpy as np
import matplotlib.pylab as plt
import obspy as obs
import pandas as pd
from obspy.signal.cross_correlation import _pad_zeros
from obspy.core.utcdatetime import UTCDateTime
from obspy.core.trace import Trace
from scipy.signal import find_peaks,detrend
from datetime import datetime
from scipy.integrate import quad,trapz
import os


""" 

RAW SPECTRUM 

For each station, raw spectrum is obtained; processing includes 
only detrending


"""

### Importing SAC files ###

file_dir = os.path.abspath(os.getcwd()) 
data_folder_dir =  os.path.join(file_dir,"data")
data_files_list = os.listdir(data_folder_dir)
stations_name = ['AC07','ACH1','ACH2','ACUE','ALJ1','APLA','ARNL','GYKA']

### Defining event origin time ###


t0 = '2023-03-18 17:12:53.2'
event_origin = datetime.strptime(t0, "%Y-%m-%d %H:%M:%S.%f")
t0 = event_origin.isoformat()                     #ISO format
t0 = UTCDateTime(t0)                              # method to convert date into compatible obspy fotmat



def raw_spectrum_subplot(files_list):  ### change from -3 secons before p-arrival and 3 seconds after coda wave lapse time
    n = len(files_list)//3    
    fig,axes = plt.subplots(n,3,figsize=(10,14))
    for i in range(0,len(files_list),3):
        for j in range(3):
            trace = obs.read('data/'+files_list[i+j])
            data = trace[0]
            station_name  = data.stats.station
            station_channel = data.stats.channel
            station_start = data.stats.starttime 
            station_end = data.stats.endtime
            npts = data.stats.npts
            dt = data.stats.delta
            detrended_data = detrend(data,type='linear')
            raw_fft = np.fft.rfft(detrended_data)
            power_spec = (np.abs(raw_fft))**2
            freqs = np.fft.rfftfreq(npts,dt)      
            ## Plotting ##    
            axes[i//3][j].plot(freqs,power_spec,linewidth=0.4)
            axes[i//3][j].set_xlim(0, 3)
            axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
            axes[i//3][j].tick_params(axis='both',labelsize=6)
            axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
    return fig, axes

### Plotting spectra per each subset ###

fig,axes = raw_spectrum_subplot(data_files_list)                                   # change input
fig.suptitle('Event: igepn2023fkei Time: {}'.format(t0),fontsize=8)
fig.supxlabel(r'f')
fig.supylabel(r'Power Spectrum')
fig.tight_layout(pad=1.25)
fig.savefig("images/raw_spectrum/all_raw_spectrum_linear_scale_0_3.jpeg",dpi=600)  # change input


""" 
Setting times 

"""

### Importing P-wave and S-wave arrival time ###
cols = ['sta','net','dist','azi','phase','time','res','n','wt','stas'] 
df = pd.read_fwf('times.txt',header=None,
                 names=cols)
df.index = df['sta']
rows = df.loc[df['sta'].isin(stations_name)][['sta','dist','azi','phase','time']]

stations_data = dict()

for station in stations_name:
    if station=='APLA':
         stations_data[station]= dict()
         stations_data[station]['distance'] = float(rows.loc[station,'dist'])
         stations_data[station]['azimuth'] = float(rows.loc[station,'azi'])
         stations_data[station]['P-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+rows.loc[station,'time'], "%Y-%m-%d %H:%M:%S.%f").isoformat())
    else:    
        stations_data[station]= dict()
        stations_data[station]['distance'] = float(rows.loc[station,'dist'][0])
        stations_data[station]['azimuth'] = float(rows.loc[station,'azi'][0])
        stations_data[station]['P-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+rows.loc[station,'time'][0],"%Y-%m-%d %H:%M:%S.%f").isoformat())
        stations_data[station]['S-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+rows.loc[station,'time'][1],"%Y-%m-%d %H:%M:%S.%f").isoformat())




def noise_spectrum_subplot(files_list,origin_time,dict_info):
    n = len(files_list)//3    
    fig,axes = plt.subplots(n,3,figsize=(10,14))
    for i in range(0,len(files_list),3):
        for j in range(3):
            trace = obs.read('data/'+files_list[i+j])
            trace.trim(origin_time,dict_info[files_list[i+j][3:7]]['P-arrival time'])
            data = trace[0]
            station_name  = data.stats.station
            station_channel = data.stats.channel
            npts = data.stats.npts
            dt = data.stats.delta
            detrended_data = detrend(data,type='linear')
            raw_fft = np.fft.rfft(detrended_data)
            power_spec = (np.abs(raw_fft))**2
            freqs = np.fft.rfftfreq(npts,dt)      
            ## Plotting ##    
            axes[i//3][j].plot(freqs,power_spec,linewidth=0.4)
            axes[i//3][j].set_xlim(0, 5)
            axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
            axes[i//3][j].tick_params(axis='both',labelsize=7)
            axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
    return fig, axes

### Plotting spectra per each subset ###

fig,axes = noise_spectrum_subplot(data_files_list,t0,stations_data)                                   # change input
fig.suptitle('Event: igepn2023fkei Time: {} \n Data: Noise prior to P-wave arrival'.format(t0),fontsize=10)
fig.supxlabel(r'f')
fig.supylabel(r'Power Spectrum')
fig.tight_layout(pad=1.25)
fig.savefig("images/prior_noise_spectrum/all_noise_spectrum_linear_scale_0_5.jpeg",dpi=600)  # change input




def p_wave_spectrum_subplot(files_list,dict_info):
    n = len(files_list)//3    
    fig,axes = plt.subplots(n,3,figsize=(10,14))
    for i in range(0,len(files_list),3):
        for j in range(3):
            if files_list[i+j][3:7]=='APLA':   # pick S wave arrival for 'APLA'
                continue
            else:
                
                trace = obs.read('data/'+files_list[i+j])
                trace.trim(dict_info[files_list[i+j][3:7]]['P-arrival time'],dict_info[files_list[i+j][3:7]]['S-arrival time'])
                data = trace[0]
                station_name  = data.stats.station
                station_channel = data.stats.channel
                npts = data.stats.npts
                dt = data.stats.delta
                detrended_data = detrend(data,type='linear')
                raw_fft = np.fft.rfft(detrended_data)
                power_spec = (np.abs(raw_fft))**2
                freqs = np.fft.rfftfreq(npts,dt)      
                ## Plotting ##    
                axes[i//3][j].plot(freqs,power_spec,linewidth=0.5)
                axes[i//3][j].set_xlim(0, 5)
                axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i//3][j].tick_params(axis='both',labelsize=7)
                axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
    return fig, axes

### Plotting spectra per each subset ###

fig,axes = p_wave_spectrum_subplot(data_files_list,stations_data)                                   # change input
fig.suptitle('Event: igepn2023fkei Time: {} \n Data: P-wave lapse times'.format(t0))
fig.supxlabel(r'f')
fig.supylabel(r'Power Spectrum')
fig.tight_layout(pad=1.25)
fig.savefig("images/p_wave_spectrum/all_p_wave_spectrum_linear_scale_0_5.jpeg",dpi=600)  # change input


def s_wave_spectrum_subplot(files_list,origin_time,dict_info):
    n = len(files_list)//3    
    fig,axes = plt.subplots(n,3,figsize=(10,14))
    for i in range(0,len(files_list),3):
        for j in range(3):
            if files_list[i+j][3:7]=='APLA':   # pick S wave arrival for 'APLA'
                continue
            else:
                dt_s_wave = dict_info[files_list[i+j][3:7]]['S-arrival time'] - origin_time
                trace = obs.read('data/'+files_list[i+j])
                trace.trim(dict_info[files_list[i+j][3:7]]['S-arrival time'],dict_info[files_list[i+j][3:7]]['S-arrival time']+2*dt_s_wave)  # s-wave windows lasts 2 times its arrival time
                data = trace[0]
                station_name  = data.stats.station
                station_channel = data.stats.channel
                npts = data.stats.npts
                dt = data.stats.delta
                detrended_data = detrend(data,type='linear')
                raw_fft = np.fft.rfft(detrended_data)
                power_spec = (np.abs(raw_fft))**2
                freqs = np.fft.rfftfreq(npts,dt)      
                ## Plotting ##    
                axes[i//3][j].plot(freqs,power_spec,linewidth=0.5)
                axes[i//3][j].set_xlim(0, 5)
                axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i//3][j].tick_params(axis='both',labelsize=6)
                axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
    return fig, axes

### Plotting spectra per each subset ###

fig,axes = s_wave_spectrum_subplot(data_files_list,t0,stations_data)                                   # change input
fig.suptitle('Event: igepn2023fkei Time: {} \n Data: S-wave lapse times'.format(t0))
fig.supylabel(r'Power Spectrum')
fig.supxlabel(r'f')
fig.tight_layout(pad=1.25)
fig.savefig("images/s_wave_spectrum/all_s_wave_spectrum_linear_scale_0_5.jpeg",dpi=600)  # change input


def coda_wave_spectrum_subplot(files_list,origin_time,dict_info):
    n = len(files_list)//3    
    fig,axes = plt.subplots(n,3,figsize=(10,14))
    for i in range(0,len(files_list),3):
        for j in range(3):
            if files_list[i+j][3:7]=='APLA':   # pick S wave arrival for 'APLA'
                continue
            else:
                dt_s_wave = dict_info[files_list[i+j][3:7]]['S-arrival time'] - origin_time
                trace = obs.read('data/'+files_list[i+j])
                trace.trim(dict_info[files_list[i+j][3:7]]['S-arrival time']+2*dt_s_wave,dict_info[files_list[i+j][3:7]]['S-arrival time']+4*dt_s_wave)  # s-wave windows lasts 2 times its arrival time
                data = trace[0]
                station_name  = data.stats.station
                station_channel = data.stats.channel
                npts = data.stats.npts
                dt = data.stats.delta
                detrended_data = detrend(data,type='linear')
                raw_fft = np.fft.rfft(detrended_data)
                power_spec = (np.abs(raw_fft))**2
                freqs = np.fft.rfftfreq(npts,dt)      
                ## Plotting ##    
                axes[i//3][j].plot(freqs,power_spec,linewidth=0.5)
                axes[i//3][j].set_xlim(0, 5)
                axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i//3][j].tick_params(axis='both',labelsize=6)
                axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
    return fig, axes

### Plotting spectra per each subset ###

fig,axes = coda_wave_spectrum_subplot(data_files_list,t0,stations_data)                                   # change input
fig.suptitle('Event: igepn2023fkei Time: {} \n Data: Coda wave lapse times'.format(t0))
fig.supylabel(r'Power Spectrum')
fig.supxlabel(r'f')
fig.tight_layout(pad=1.25)
fig.savefig("images/coda_wave_spectrum/all_coda_wave_spectrum_linear_scale_0_5.jpeg",dpi=600) 





""" Effective duration by Husid (1969) """

from scipy.integrate import trapezoid,simpson
def integrand_intensity(a):
    return a**2
f = lambda a,i:(a[i])**2

def intensity_trapz(y,x,i):
    return trapezoid(y[:i],x[:i])/trapezoid(y,x)

def intensity_simps(y,x,i):
    return simpson(y[:i],x[:i])/simpson(y,x)
# def quad_intensity(a,i,t):
#     return quad(integrand_intensity,0,t[i],args=(a))/quad(integrand_intensity,0,t[len(t)-1],args=(a))

data = obs.read('data/'+data_files_list[1])
data = data[0]
dt = data.stats.delta
times = data.times()



I_simps = [intensity_simps([d**2 for d in data],times,i) for i in range(1,len(times))]
plt.plot(I_simps,'b-')
plt.xlim(0,150)
plt.show()


def intensity_subplot(files_list,origin_time,dict_info):
    n = len(files_list)//3    
    fig,axes = plt.subplots(n,3,figsize=(10,14))
    for i in range(0,len(files_list),3):
        for j in range(3):
            trace = obs.read('data/'+files_list[i+j])
            trace.trim(origin_time,dict_info[files_list[i+j][3:7]]['P-arrival time'])
            data = trace[0]
            times = data.times()
            station_name  = data.stats.station
            station_channel = data.stats.channel
            npts = data.stats.npts
            dt = data.stats.delta
            detrended_data = detrend(data,type='linear')
            I_simps = [intensity_simps([d**2 for d in detrended_data],times,i) for i in range(1,len(times),100)]   
            time = [times[i] for i in range(1,len(times),100)]
            ## Plotting ##    
            axes[i//3][j].plot(time,I_simps,linewidth=0.4)
            axes[i//3][j].set_xlim(0, 120)
            axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
            axes[i//3][j].tick_params(axis='both',labelsize=7)
            axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
    return fig, axes

### Plotting spectra per each subset ###

fig,axes = noise_spectrum_subplot(data_files_list,t0,stations_data)                                   # change input
fig.suptitle('Event: igepn2023fkei Time: {} \n Data: Noise prior to P-wave arrival'.format(t0),fontsize=10)
fig.supxlabel(r'f')
fig.supylabel(r'Power Spectrum')
fig.tight_layout(pad=1.25)
fig.savefig("images/Intensity/all_noise_spectrum_linear_scale_0_5.jpeg",dpi=600)  # change input
