
import numpy as np
from scipy.signal import find_peaks,detrend
import obspy as obs


def fft_raw_data(data,N:int,dt:float,station:str,channel='xyz'):
    """
    Parameters
    ----------
    data : array/list
        obtained from mseed file [0]
    N : int
        number of points.
    dt : float
        sampling time interval.
    station : str
        sensor label.
    channel : str
        direction/channel.

    Returns
    -------
    ax : Plot containing power spectrum of raw data.

    """
    detrended_data = detrend(data,type='linear')
    fourier = np.fft.rfft(detrended_data)
    freqs = np.fft.rfftfreq(N,dt)

    return [fourier,freqs]

def raw_HV(x_data,y_data,z_data,N:int,dt:float,station="TBD"):
    fx = fft_raw_data(x_data, N, dt, station)[0]
    fy = fft_raw_data(y_data, N, dt, station)[0]
    fz = fft_raw_data(z_data, N, dt, station)[0]
    freqs = fft_raw_data(z_data, N, dt, station)[1]
    fx_amp = np.abs(fft_raw_data(x_data, N, dt, station)[0])
    fy_amp = np.abs(fft_raw_data(y_data, N, dt, station)[0])
    fz_amp = np.abs(fft_raw_data(z_data, N, dt, station)[0])
    
    sqr_av =  (np.sqrt(fx_amp**2+fy_amp**2)/2)/fz_amp
    
    return [sqr_av,freqs]
    
    

class Signal:
    def  __init__(self,dat1,dt,dat2=[],dat3=[],min_val=5500,max_val=12000,
                  min_ratio=0.2,max_ratio=2.5,dt_window=200):
        self.dat1 = dat1
        self.dat2 = dat2
        self.dat3 = dat3
        self.dt = dt
        self.dt_window = dt_window
        self.min_val = min_val
        self.max_val = max_val
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        # self.HV_processing_peaks()
        self.HV_processing_sta_lta()
        # self.HV_processing_sta_lta_eff()
    
    # def HV_processing_peaks(self):
    #     HV = []
    #     freqs = []
    #     windows_x=self.dat1.slide(window_length=self.dt_window,step=self.dt_window,include_partial_windows=False)
    #     windows_y=self.dat2.slide(window_length=self.dt_window,step=self.dt_window,include_partial_windows=False)
    #     windows_z=self.dat3.slide(window_length=self.dt_window,step=self.dt_window,include_partial_windows=False)
        
    #     for i,(winx,winy,winz) in enumerate(zip(windows_x,windows_y,windows_z)):
    #         detrended_data_x = detrend(winx,type='linear')
    #         detrended_data_y = detrend(winy, type='linear')
    #         detrended_data_z = detrend(winz, type='linear')
    #         bool1 = self.min_val < max(detrended_data_x) < self.max_val
    #         bool2 = self.min_val < max(detrended_data_y) < self.max_val
    #         bool3 = self.min_val < max(detrended_data_z) < self.max_val
    #         if bool1 or bool2 or bool3:
    #             continue
    #         m =len(detrended_data_x)
    #         tapered_data_x=np.hanning(m)*detrended_data_x
    #         tapered_data_y=np.hanning(m)*detrended_data_y
    #         tapered_data_z=np.hanning(m)*detrended_data_z
    #         HV.append(raw_HV(tapered_data_x,tapered_data_y,tapered_data_z,m,self.dt)[0])
    #         freqs.append(raw_HV(tapered_data_x,tapered_data_y,tapered_data_z,m,self.dt)[1])      
    #     self.freqs_peaks = freqs
    #     self.hv_peaks = HV
        
    def HV_processing_sta_lta(self):
        HV = []
        freqs = []
        windows_x=self.dat1.slide(window_length=self.dt_window,step=self.dt_window,include_partial_windows=False)
        windows_y=self.dat2.slide(window_length=self.dt_window,step=self.dt_window,include_partial_windows=False)
        windows_z=self.dat3.slide(window_length=self.dt_window,step=self.dt_window,include_partial_windows=False)
        df = self.dat1.stats.sampling_rate
        for i,(winx,winy,winz) in enumerate(zip(windows_x,windows_y,windows_z)):
            detrended_data_x = detrend(winx,type='linear')
            detrended_data_y = detrend(winy, type='linear')
            detrended_data_z = detrend(winz, type='linear')  
            cft_x = classic_sta_lta(detrended_data_x,int(df*1), int(df*30))
            cft_y = classic_sta_lta(detrended_data_y,int(df*1), int(df*30))
            cft_z = classic_sta_lta(detrended_data_z,int(df*1), int(df*30))
            bool1 = self.min_ratio < max(cft_x) < self.max_ratio
            bool2 = self.min_ratio < max(cft_y) < self.max_ratio
            bool3 = self.min_ratio < max(cft_z) < self.max_ratio
            if bool1 and bool2 and bool3:
                m =len(detrended_data_x)
                tapered_data_x=np.hanning(m)*detrended_data_x
                tapered_data_y=np.hanning(m)*detrended_data_y
                tapered_data_z=np.hanning(m)*detrended_data_z
                HV.append(raw_HV(tapered_data_x,tapered_data_y,tapered_data_z,m,self.dt)[0])
                freqs.append(raw_HV(tapered_data_x,tapered_data_y,tapered_data_z,m,self.dt)[1])      
        self.freqs_sta_lta = freqs
        self.hv_sta_lta = HV
        
        

            