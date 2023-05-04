# -*- coding: utf-8 -*-
"""
Created on Thu May  4 15:54:00 2023

@author: fotogrametria
"""


from math import log, ceil, floor
def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return min(possible_results, key= lambda z: abs(x-2**z))


import obspy as obs
from scipy.signal import detrend,cosine
from obspy.core.trace import Trace
import numpy as np

dict_hist = dict()
tr = obs.read('data/EC.ISPG..HNE_20230318T121305.SAC')
for windowed_tr in tr.slide(window_length=3600,step=1800):
    for sub_windowed_tr in windowed_tr.slide(window_length=3600/13,step=int(0.75*3600/13)):
        nw = len(sub_windowed_tr[0])
        n2 = 2**(closest_power(nw)-1)    
        data = sub_windowed_tr[0][:n2]
        dt = sub_windowed_tr[0].stats.delta
        detrended = detrend(data)
        taper = cosine(n2) 
        processed_data = taper*detrended 
        fft = np.fft.rfft(processed_data)
        psd = np.abs(fft)**2
        
        ### psd without zero frequency
        psd = psd[1:]
        freqs = np.fft.rfftfreq(n2,dt)
        periods = np.reciprocal(freqs[1:])
        
        #reversing order to have increasing order in period
        psd = np.flip(psd)
        periods = np.flip(periods)
        
        dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)] = {'Central_period':[],'Average_PSD':[]}

        for i in range(len(periods)-8):
            dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)]['Central_period'].append(np.sqrt(periods[i]*periods[i+8]))
                                                                                                                                   
            dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)]['Average_PSD'].append(np.mean(psd[i:i+8]))
            

keys = list(dict_hist.keys())
for key in keys:
    dict_hist[key]['Average_PSD_dB'] = 10*np.log10(np.array(dict_hist[key]['Average_PSD']))

            
            
            
            
            
        
        
 

def periods_octave(p0,max):
    p=p0
    periods = [p]
    while p<max:
        p*=2**(1/8)
        periods.append(p)
    return periods
        
        

        
        
        
#                 npts = data.stats.npts
#                 dt = data.stats.delta
#                 detrended_data = detrend(data,type='linear')
#                 raw_fft = np.fft.rfft(detrended_data)
#                 power_spec = (np.abs(raw_fft))**2
#                 freqs = np.fft.rfftfreq(npts,dt)   
    
