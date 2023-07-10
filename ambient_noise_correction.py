# -*- coding: utf-8 -*-
"""
Created on Tue May  9 17:02:11 2023

@author: fotogrametria
"""

#TODO: use nextpow2 from either scipy or obspy or numpy



# from obspy.clients.iris import Client
# from obspy import UTCDateTime
# dt1 = UTCDateTime("2018-01-01T00:00:00")
# dt2 = UTCDateTime("2018-01-02T0:00:00")
# client = Client()
 
# st1 = client.timeseries("IU", "OTAV", "00", "BHZ", dt1,dt2,
#     filter=["correct", "demean", "lp=2.0","units=ACC"])


# dt3 = UTCDateTime("2018-01-02T00:00:00")
# dt4 = UTCDateTime("2018-01-03T00:00:00")
# client = Client()
 
# st2 = client.timeseries("IU", "OTAV", "00", "BHZ", dt2,dt3,
#     filter=["correct", "demean", "lp=2.0","units=ACC"])
# st3 = client.timeseries("IU", "OTAV", "00", "BHZ", dt3,dt4,
#     filter=["correct", "demean", "lp=2.0","units=ACC"])


# st = st1 + st2 
# st = st + st3


from math import log, ceil, floor
def closest_power(x):
    possible_results = floor(log(x, 2)), ceil(log(x, 2))
    return min(possible_results, key= lambda z: abs(x-2**z))


from obspy.signal import PPSD         
import matplotlib.pyplot as plt            
from obspy.core.stream import Stream
import obspy as obs
from scipy import interpolate
from scipy.signal import detrend,cosine,welch
from obspy.core.trace import Trace
import numpy as np
import os


# st = obs.read('D:/Proyectos/Earthquake_version/Earthquake_March18/data_ISPG/BW.KW1..EHZ.D.2011.037')
# tr = st.select(id="BW.KW1..EHZ")[0]
# inv = obs.read_inventory("D:/Proyectos/Earthquake_version/Earthquake_March18/data_ISPG/BW_KW1.xml")



# ppsd = PPSD(tr.stats, metadata=inv)
# ppsd.add(st)

# ppsd.plot()
# ppsd.plot('Obspy_PSD.jpeg')
# tr.remove_response(inventory=inv,output='ACC') 


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
# tr = obs.read('data/6month.mseed')
# x = tr[0]
# data = x.data
# dt = x.stats.delta
# f = x.stats.sampling_rate


# data = [data,data]
# psd = welch(data,f,nperseg=600,noverlap=300,detrend='linear')
# plt.plot(psd[0],psd[1])


# for windowed_tr in tr.slide(window_length=600,step=300):
#     x = windowed_tr[0]
#     data = x.data
#     welch_out = welch(data,f,nperseg=60,noverlap=30,nfft=2**12,detrend='linear')
#     freq = welch_out[0]
#     psd = welch_out[1]
#     plt.loglog(1./freq,psd)

# plt.show()

def signal_processing(stream,conversion):
    nw = len(stream[0].data)
    n2 = 2**(closest_power(nw)-1)    
    data = stream[0].data[:n2]*(1/conversion)  # conversion from nm/s**2 to m/s**2
    dt = stream[0].stats.delta
    detrended = detrend(data)
    taper = cosine(n2) 
    processed_data = taper*detrended 
    fft = np.fft.rfft(processed_data)
    psd = ((np.abs(fft))**2)*(2*dt/n2)  # added a normalization factor based on p.4 from McNamara
    psd = psd[1:]                     # removed zero value 
    freqs = np.fft.rfftfreq(n2,dt)
    periods = np.reciprocal(freqs[1:])
    
    #reversing order to have increasing order in period
    psd = np.flip(psd)
    periods = np.flip(periods)       
    dict_info = {'psd':psd,'periods':periods}
    
    return dict_info
    


dict_hist = dict()

folder_dir ='D:/Proyectos/Earthquake_version/Earthquake_March18/data_ISPG' 
file_dir = os.listdir(folder_dir)
first_file_dir = os.path.join(folder_dir,file_dir[0])
stream = obs.read(first_file_dir)
for file in file_dir:
    file_dir = os.path.join(folder_dir,file)
    stream += obs.read(file_dir)
stream.merge(method=1,fill_value=0)



# tr = obs.read('D:/Proyectos/Earthquake_version/Earthquake_March18/data_ISPG/EC.ZALD..HNZ.2015.182')

# st = obs.read('data/EC.ISPG..HHE_20230318T121305.SAC')

for windowed_tr in stream.slide(window_length=600,step=300):
    psd_hour = [] 
    for sub_windowed_tr in windowed_tr.slide(window_length=600/4,step=int(0.25*600/4)):
        result = signal_processing(sub_windowed_tr,1e9)
        periods = result['periods']
        psd = result['psd']
        psd_hour.append(psd)
        
        # T0 = periods[0]
        # Ts = [octave(T0,i) for i in range(int(8*np.log2(max(periods)/T0))-1)]
        # Ts_Tl = [(find_closest(periods, ts)[1],find_closest(periods, 2*ts)[1]) for ts in Ts ][:-5]
        # indeces = [(find_closest(periods, ts)[0],find_closest(periods, 2*ts)[0]) for ts in Ts ][:-5]         

        
        # dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)] = {'Central_period':[],'Average_PSD':[]}

        # for i in range(len(periods)-8):
        #     dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)]['Central_period'].append(np.sqrt(periods[i]*periods[i+8]))
                                                                                                                                   
        #     dict_hist[str(sub_windowed_tr[0].stats.starttime)+'-'+ str(sub_windowed_tr[0].stats.endtime)]['Average_PSD'].append(np.mean(psd[i:i+8]))

    dict_hist[str(windowed_tr[0].stats.starttime)] = {'Central_period':[],'Hourly_average_PSD':[],'Interpolation_PSD':[],
                                                      'Interpolated_hourly_average_PSD':[],'Full_octave_average_PSD':[]}
    
    periods_interpS = np.logspace(np.log2(periods[0]),np.log2(max(periods)),int((1/0.125)*np.log2(max(periods))),base=2)
    
    periods_interpL = 2*periods_interpS
    
    merge_periods0 = np.concatenate((periods,periods_interpS))
    
    merge_periods = np.sort(np.concatenate((merge_periods0,periods_interpL)))
    
    dict_hist[str(windowed_tr[0].stats.starttime)]['Hourly_average_PSD'].append(np.mean(psd_hour,axis=0)) 
    
    f = interpolate.interp1d(periods,dict_hist[str(windowed_tr[0].stats.starttime)]['Hourly_average_PSD'][0],fill_value="extrapolate")
    
    dict_hist[str(windowed_tr[0].stats.starttime)]['Interpolated_hourly_average_PSD'] = f(merge_periods)    
    
    indeces =[(find_closest(merge_periods,ts)[0],find_closest(merge_periods, 2*ts)[0]) for ts in periods_interpS]         
    for tup in indeces[:-2]:
        dict_hist[str(windowed_tr[0].stats.starttime)]['Central_period'].append(np.sqrt(merge_periods[tup[0]]*merge_periods[tup[1]]))
        dict_hist[str(windowed_tr[0].stats.starttime)]['Full_octave_average_PSD'].append(np.mean(dict_hist[str(windowed_tr[0].stats.starttime)]['Interpolated_hourly_average_PSD'][tup[0]:tup[1]]))

keys = list(dict_hist.keys())
for key in keys:
    dict_hist[key]['Average_PSD_dB'] = 10*np.log10(np.array(dict_hist[key]['Full_octave_average_PSD']))




AHNM = np.array([-91.5,-91.5,-97.41,-110.50,-120.00,-98.00,-96.5,-101.0,-105.0,-91.25])
pHNM = np.array([0.01,0.1,0.22,0.32,0.80,3.80,4.6,6.3,7.10,150.0])

AHNM0 = np.flip(AHNM)
pHNM0 =  1/np.flip(pHNM)

ALNM = np.array([-135.0,-135.0,-130.0,-118.25])
pLNM = np.array([0.01,1,10,150])

ALNM0 = np.flip(ALNM)
pLNM0 =  1/np.flip(pLNM)


PDFs = []
HIST = []
period =  dict_hist[key]['Central_period']
periods_order = np.flip(np.array(period))
frequency = 1/periods_order

for j in range(len(dict_hist[key]['Central_period'])):
    hist = [dict_hist[key]['Average_PSD_dB'][len(dict_hist[key]['Central_period'])-j-1] for key in keys] 
    histogram = np.histogram(np.array(hist),bins=np.arange(-200,-80,1),density=True)
    plt.plot(histogram[1][:-1],histogram[0])
    HIST.append(histogram)
    PDFs.append(histogram[0])
plt.show()        
    
db = histogram[1][1:]          
PDF = np.transpose(PDFs)    
T,dB = np.meshgrid(frequency,db)                        
h = plt.contourf(T, dB, PDF)



plt.semilogx(pLNM0,ALNM0,label='ALNM',color = 'skyblue')
plt.semilogx(pHNM0,AHNM0,label='AHNM',color = 'red')
plt.legend(loc='lower left')
plt.xscale('log')
plt.xlim(1/(0.8e2),1/(3e-2)) # limits adjusted manually 
plt.colorbar()
plt.grid(True,which="both",linewidth = "0.5")
plt.xlabel('Frequency [s]')
plt.ylabel('Amplitude [$m^{2}$/$s^{4}$/Hz]')
plt.title('ISPG HNZ')
plt.savefig('ISPG_HNZ_Cauzzi_noise_model.jpeg',dpi=600)
plt.show()                
        
        
        


        