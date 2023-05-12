# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:02:11 2023

@author: fotogrametria
"""

#TODO: use nextpow2 from either scipy or obspy or numpy

from math import log, ceil, floor
def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return min(possible_results, key= lambda z: abs(x-2**z))



            
            
from obspy.core.stream import Stream
import obspy as obs
from scipy import interpolate
from scipy.signal import detrend,cosine,welch
from obspy.core.trace import Trace
import numpy as np
import os



def octave(T0,n):
    if n==0:
        return T0
    else:
         return 2**(0.125)*octave(T0,n-1)
    
       
                    
def find_closest(arr, val):
       idx = np.abs(arr - val).argmin()
       return (idx,arr[idx])

                

            

# strea = Stream()
# main_dir = os.path.abspath(os.getcwd()) 
# data_folder_dir =  os.path.join(main_dir,"201506")
# data_files_list = os.listdir(data_folder_dir)

# traces = []
# for path in data_files_list[:40]:
#     file_path = os.path.join(data_folder_dir,path)
#     file = obs.read(file_path,format='GCF')
#     file0 = file[0]
#     file0.stats.network = 'EC'
#     file0.stats.station = 'ZALD'
#     if file0.stats.sampling_rate == 100.0 and file0.stats.channel[2:3] == 'E':
#         file0.stats.channel = 'HNE'
#         traces.append(file0)
#         sti = Stream(traces=file0)
#         strea += sti

# strea.merge()
# tr  = strea





dict_hist = dict()
tr = obs.read('data/EC.ISPG..HNE_20230318T121305.SAC')
x = tr[0]
data = x.data
dt = x.stats.delta
f = x.stats.sampling_rate

psd = welch(data,f,nperseg=600,noverlap=300,detrend='linear',)



dict_hist = dict()
tr = obs.read('data/EC.ISPG..HNE_20230318T121305.SAC')
for windowed_tr in tr.slide(window_length=600,step=300):
    psd_hour = [] 
    for sub_windowed_tr in windowed_tr.slide(window_length=600/4,step=int(0.25*600/4)):
        nw = len(sub_windowed_tr[0])
        n2 = 2**(closest_power(nw)-1)    
        data = sub_windowed_tr[0][:n2]
        dt = sub_windowed_tr[0].stats.delta
        detrended = detrend(data)
        taper = cosine(n2) 
        processed_data = taper*detrended 
        fft = np.fft.rfft(processed_data)
        psd = (np.abs(fft)**2)*(2*dt/n2) 
        
        ### psd without zero frequency
        psd = psd[1:]   #added a normalization factor based on p.4 from McNamara
        freqs = np.fft.rfftfreq(n2,dt)
        periods = np.reciprocal(freqs[1:])
        
        #reversing order to have increasing order in period
        psd = np.flip(psd)
        periods = np.flip(periods)
        
        psd_hour.append(psd)
        
        # T0 = periods[0]
        # Ts = [octave(T0,i) for i in range(int(8*np.log2(max(periods)/T0))-1)]
        # Ts_Tl = [(find_closest(periods, ts)[1],find_closest(periods, 2*ts)[1]) for ts in Ts ][:-5]
        # indeces = [(find_closest(periods, ts)[0],find_closest(periods, 2*ts)[0]) for ts in Ts ][:-5]         

        
        # dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)] = {'Central_period':[],'Average_PSD':[]}

        # for i in range(len(periods)-8):
        #     dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)]['Central_period'].append(np.sqrt(periods[i]*periods[i+8]))
                                                                                                                                   
        #     dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)]['Average_PSD'].append(np.mean(psd[i:i+8]))

    dict_hist[str(windowed_tr[0].stats.starttime)] = {'Central_period':[],'Hourly_average_PSD':[],'Interpolation_PSD':[],'Interpolated_hourly_average_PSD':[],'Full_octave_average_PSD':[]}
    periods_interpS = np.logspace(np.log10(periods[0]),np.log10(max(periods)),500)
    periods_interpL = 2*periods_interpS
    merge_periods0 = np.concatenate((periods,periods_interpS))
    merge_periods = np.sort(np.concatenate((merge_periods0,periods_interpL)))
    dict_hist[str(windowed_tr[0].stats.starttime)]['Hourly_average_PSD'].append(np.mean(psd_hour,axis=0)) 
    f = interpolate.interp1d(dict_hist[str(windowed_tr[0].stats.starttime)]['Hourly_average_PSD'][0],periods,fill_value="extrapolate")
    dict_hist[str(windowed_tr[0].stats.starttime)]['Interpolated_hourly_average_PSD'] = f(merge_periods)    
    indeces =[(find_closest(merge_periods,ts)[0],find_closest(merge_periods, 2*ts)[0]) for ts in periods_interpS]         
    for tup in indeces:
        dict_hist[str(windowed_tr[0].stats.starttime)]['Central_period'].append(np.sqrt(merge_periods[tup[0]]*merge_periods[tup[1]]))
        dict_hist[str(windowed_tr[0].stats.starttime)]['Full_octave_average_PSD'].append(np.mean(dict_hist[str(windowed_tr[0].stats.starttime)]['Hourly_average_PSD'][0][tup[0]:tup[1]]))

keys = list(dict_hist.keys())
for key in keys:
    dict_hist[key]['Average_PSD_dB'] = 10*np.log10(np.array(dict_hist[key]['Full_octave_average_PSD']))

import matplotlib.pyplot as plt


AHNM = [-91.5,-91.5,-97.41,-110.50,-120.00,-98.00,-96.5,-101.0,-105.0,-91.25]
pAHNM = [0.01,0.1,0.22,0.32,0.80,3.80,4.6,6.3,7.10,150.0]

ALNM = [-135.0,-135.0,-130.0,-118.25]
pLNM = [0.01,1,10,150]

PDFs = []
HIST = []
period =  dict_hist[key]['Central_period']

for j in range(len(dict_hist[key]['Central_period'])):
    hist = [dict_hist[key]['Average_PSD_dB'][j] for key in keys] 
    histogram = np.histogram(np.array(hist),bins=np.arange(-250,0,1),density=True)
    HIST.append(histogram)
    PDFs.append(histogram[0])
        
    
db = histogram[1][1:]
          
PDF = np.transpose(PDFs)    
  
T,dB = np.meshgrid(period,db)        
                
h = plt.contourf(T, dB, PDF)



plt.semilogx(pLNM,ALNM,label='ALNM')
plt.semilogx(pAHNM,AHNM,label='AHNM')
plt.legend()
plt.xscale('log')
plt.colorbar()

plt.show()                
        
        


        