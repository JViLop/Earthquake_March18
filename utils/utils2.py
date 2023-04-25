# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 18:32:34 2023

@author: joanv
"""

import csv
import numpy as np
from scipy.signal import find_peaks,detrend
import obspy as obs
from obspy.signal.filter import bandpass,envelope
from obspy.signal.util import smooth
import os
import shutil
from scipy.integrate import trapezoid,simpson
import matplotlib.pylab as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from datetime import datetime
from obspy.core.utcdatetime import UTCDateTime
import re





"""

CODA-Q ANALYSIS

"""
def coda_analysis_spectrum_plots(main_dir,files_set,stations_data,t0,fc,plot_type='Coda_Q',dt_adjust = 0,factor_s = 2, factor_e = 4):
    plots_dir = os.path.join(main_dir,"graphs")
    plot_dir = os.path.join(plots_dir,plot_type)
    csv_dir = os.path.join(main_dir,'csv_files')
    csv_dir = os.path.join(csv_dir,plot_type)
    
    if fc==0.75:
        a = 1
    elif fc==1.5:
        a = 2 
    elif fc==3:
        a = 3
    elif fc==6:
        a = 4
    elif fc==12:
        a = 5
    else:
        a = 6
        
    csv_file_dir = os.path.join(csv_dir,str(a)+'_'+plot_type +"_f_"+str(fc)+"_"+"factor_dt_end_"+str(factor_e)+"_.csv")
    file_dir = os.path.join(plot_dir,str(a)+ '_'+plot_type +"_f_"+str(fc)+"_"+"factor_dt_end_"+str(factor_e)+"_.jpeg")
    # file_dir1 = os.path.join(plot_dir,plot_type + "_attenuation"+"_f_"+str(fc)+"_"+"tc_end_"+str(factor_e)+"_SMOOTH.jpeg")
    if not os.path.isdir(plot_dir) and not os.path.isdir(csv_dir) :    
        os.makedirs(plot_dir)    
        os.makedirs(csv_dir)
    
    def coda_analysis_spectrum_subplot(files_list,origin_time,dict_info,f,factor_start = 2,factor_end = 4):
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(12,16))
        # fig1,axes1 = plt.subplots(n,3,figsize=(12,16))
        dataQ = {'station':[],'E':[],'N':[],'Z':[]}
        for i in range(0,len(files_list),3):
            for j in range(3):
                dt_s_wave = dict_info[files_list[i+j][3:7]]['S-arrival time'] - origin_time
                trace = obs.read('data/'+files_list[i+j])
                trace1 = obs.read('data/'+files_list[i+j])
                trace1.trim(origin_time,origin_time+factor_end*dt_s_wave)
                data1 = trace1[0]
                t1 = data1.times()
                trace.trim(origin_time+factor_start*dt_s_wave,origin_time+factor_end*dt_s_wave)  # s-wave windows lasts 2 times its arrival time
                data = trace[0]
                lapse_time_window = factor_end*dt_s_wave -factor_start*dt_s_wave
                factor_diff = factor_end-factor_start
                time = data.times()+factor_start*dt_s_wave
                t  = np.array(time).reshape((-1,1))
                station_name  = data.stats.station
                station_channel = data.stats.channel
                df = data.stats.sampling_rate
                fmin,fmax = (2/3)*f, (4/3)*f
                detrended_data = detrend(data,type='linear')
                band_pass_data = bandpass(detrended_data,fmin,fmax,df)
                envelp = envelope(band_pass_data)
                #A = smooth(envelp,int(5/0.01))
                A = moving_ave(envelp,int(5/0.01)) ### Noisepy function
                ln_At = np.log(A*time)
                # model = np.polyfit(t,ln_At,1)
                model = LinearRegression().fit(t,ln_At)
                r_sq = model.score(t,ln_At)
                m,b = model.coef_[0],model.intercept_
                Q = -(np.pi*f)/m
                A0 = np.exp(b)
                At = A0*(t**(-1))*np.exp(-np.pi*f*t/Q)
                if j==0:
                    dataQ['station'].append(station_name)
                dataQ[station_channel[-1]].append(Q)
                
                # predict = np.poly1d(model)
                # linear_fit = predict(t)
                linear_fit = model.predict(t)
                axes[i//3][j].plot(time,linear_fit,color='k',linewidth=0.8,label="$Q_{c}=$"+ "{}".format(str(round(Q,2)))+" "+" $R^{2}=$"+ "{}".format(str(round(r_sq,3))) )
                axes[i//3][j].scatter(time,ln_At,color='b',s=0.15,linewidths=0.1,marker='+')
                axes[i//3][j].legend(fontsize=5)
                axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i//3][j].tick_params(axis='both',labelsize=6)
                axes[i//3][j].set_title(station_name + ' ' + station_channel + ' '+  '\n $\Delta t_{coda}=$'+'{}'.format(factor_diff)+'$\Delta t_{S_{arrival}}$='+'{}'.format(round(lapse_time_window,1))+'s',fontdict={'fontsize':4.5,'color':'blue'})
                # axes1[i//3][j].plot(t,At,color='k',linestyle='--',linewidth=0.5,label="$Q_{c}=$"+"{}".format(str(round(Q,2))))
                # axes1[i//3][j].plot(t1,data1,color='r',linewidth=0.1)
                # axes1[i//3][j].legend(fontsize=3)
                # axes1[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                # axes1[i//3][j].tick_params(axis='both',labelsize=6)
                # axes1[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})

        # dfQ = pd.DataFrame(dataQ)
        return fig, axes,dataQ #fig1,axes1 
    
    ### Plotting spectra per each subset ###
    
    fig,axes,dataQ= coda_analysis_spectrum_subplot(files_set,t0,stations_data,fc,factor_start=factor_s,factor_end = factor_e)                                    
    fig.suptitle('Event: igepn2023fkei Time: {} \n Linear Regression for Q factor'.format(t0))
    fig.supylabel(r'$log(A(t,f))+ log(t)$')
    fig.supxlabel(r't')
    fig.tight_layout(pad=1.25)
    fig.savefig(file_dir,dpi=600)   
    df=pd.DataFrame(dataQ,index=dataQ['station'])
    df.to_csv(csv_file_dir)                   
    # fig1.suptitle('Event: igepn2023fkei Time: {} \n Model for attetuation (Aki and Chouet)'.format(t0))
    # fig1.supylabel(r'Acceleration')
    # fig1.supxlabel(r't')
    # fig1.tight_layout(pad=1.25)
    # fig1.savefig(file_dir1,dpi=600) 




    
class Earthquake:
    
    def __init__(self,path_data_folder:str,path_event_text_file:str,path_summary_acc_text_file:str,path_summary_vel_text_file:str):
        self.main_dir = os.path.abspath(os.getcwd()) 
        self.data_set = os.listdir(path_data_folder)
        self.data_set_abs_path = [os.path.join(path_data_folder,item) for item in os.listdir(path_data_folder)]
        self.path_data_folder = path_data_folder
        self.path_event_text_file = path_event_text_file
        self.path_summary_acc_text_file = path_summary_acc_text_file  
        self.path_csv_files = os.path.join(self.main_dir,'csv_files')
        self.path_plots = os.path.join(self.main_dir,'plots')
        self.stations_name = list({file[3:7] for file in os.listdir(self.path_data_folder)})
        self.event_info()
        self.stations_info()
        # self.raw_spectrum_plots(10,'linear')
        # self.noise_spectrum_plots(10,'linear',0,0)
        self.p_wave_spectrum_plots(10, 'linear')
        # self.s_wave_spectrum_plots(10, 'linear')
        # self.coda_wave_spectrum_plots(10, 'linear')
        # self.intensity_plots(100)
    @staticmethod
    def line_index(path:str,string:str):
        with open(path, 'r') as fp:
            for l_no, line in enumerate(fp):
                # search string
                if string in line:
                    c =l_no
                    break
                    # don't look for next lines
        return c    
    @staticmethod
    def string_finder(path:str,string:str):
        with open(path, 'r') as fp:
            for l_no, line in enumerate(fp):
                # search string
                if string in line: 
                    c = line
                    break
                    # don't look for next lines
        return c
            
    def event_info(self):
        
        event_id = Earthquake.string_finder(self.path_event_text_file,'Public ID')
        event_id_regex = re.compile(r'(\w{5}\d{4}\w{4})')
        event_id = re.findall(event_id_regex,event_id)[0]
        
        ### event date and time ###
        date = Earthquake.string_finder(self.path_event_text_file,'Date') 
        date_regex = re.compile(r'(\d{4}-\d{2}-\d{2})')
        date = re.findall(date_regex,date)[0]
        
        time = Earthquake.string_finder(self.path_event_text_file,'Time') 
        time_regex = re.compile(r'(\d{2}:\d{2}:\d*\.?\d*)')
        time = re.findall(time_regex,time)[0]
        
        event_origin = datetime.strptime(date +' '+time, "%Y-%m-%d %H:%M:%S.%f")
        t0 = event_origin.isoformat()                     #ISO format
        t0 = UTCDateTime(t0)  
        
         ### event latitude,  and depth ### 
        
        latitude = Earthquake.string_finder(self.path_event_text_file,'Latitude') 
        latitude_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
        latitude = re.findall(latitude_regex,latitude)[0]
        latitude = float(latitude)
          
        longitude = Earthquake.string_finder(self.path_event_text_file,'Longitude') 
        longitude_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
        longitude = re.findall(longitude_regex,longitude)[0]
        longitude = float(longitude)
        
        depth = Earthquake.string_finder(self.path_event_text_file,'Depth') 
        depth_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
        depth = re.findall(depth_regex,depth)[0]
        depth = float(depth)
        
        self.earthquake_info = {'Event ID':event_id,' Origin Time':t0,' Latitude':latitude,' Longitude':longitude,' Depth':depth}
        self.title = ','.join(f'{k}:{v}' for k,v in self.earthquake_info.items())
        self.date = date
            

    @staticmethod
    def csv_creator(path:str,name_csv:str,line_start:int,line_end=None):
        """
    
        Parameters
        ----------
        path : str
            
        name_csv : str
            
        line_start : int
            
        line_end : TYPE, optional

        Returns
        -------
        None.

        """
        file = open(path)
        content = file.readlines()[line_start:line_end]
        content = [re.sub(r'\s+',",",line) for line in content]
        content = [re.findall('(?<=,)[^,]+(?=,)',item)[:6] for item in content][1:]   
        csv_file = open('{}'.format(name_csv), 'w+', newline ='')
        with csv_file:   
            write = csv.writer(csv_file)
            write.writerows(content)
    
    @staticmethod
    def url_finder(x,station,comp):
        pattern = fr'{station}+[^0-9]+{comp}'
        re_pattern = re.compile(pattern)
        for item in x:
            if  re.findall(re_pattern, item):
                return item
            
    def stations_info(self):
        lineStationPhase = Earthquake.line_index(self.path_event_text_file,'Phase arrivals:')  
        lineStationMagn = Earthquake.line_index(self.path_event_text_file,'Station magnitudes:')
        path_phase_arrivals = os.path.join(self.path_csv_files,'station_phase_arrivals.csv')
        path_magnitudes  = os.path.join(self.path_csv_files,'station_magnitudes.csv')
        if os.path.isdir(self.path_csv_files):
            shutil.rmtree(self.path_csv_files)
        os.mkdir(self.path_csv_files)    
        Earthquake.csv_creator(self.path_event_text_file,path_phase_arrivals,lineStationPhase,line_end=lineStationMagn) 
        Earthquake.csv_creator(self.path_event_text_file,path_magnitudes,lineStationMagn)    
        cols = ['sta','net','dist','azi','phase','time']
        df = pd.read_csv( path_phase_arrivals,names=cols,sep=',',skiprows=1)
        rows = df.loc[df['sta'].isin(self.stations_name)][cols]
        rows = rows.set_index('sta')

        cols1 = ['sta','net','dist','azi','type','value']
        df1 = pd.read_csv(path_magnitudes ,names=cols1,sep=',',skiprows=1)
        rows1 = df1.loc[df1['sta'].isin(self.stations_name)][cols1]
        rows1 = rows1.set_index('sta')

        cols2 =["NET.STATION.LOCID.CMPNM", "MaxAcc", "TimeOfMaxAcc", "DIST", "AZ", "BAZ",
                "PeakVecACC", "EstimatedDuration."]
        df2 = pd.read_csv(self.path_summary_acc_text_file,names=cols2,sep=' ',skiprows=1)
        df2['station'] = df2['NET.STATION.LOCID.CMPNM'].str[3:7]
        df2['CHN'] = df2['NET.STATION.LOCID.CMPNM'].str[-3:]
        rows2 = df2.loc[df2['station'].isin(self.stations_name)][["station","CHN","MaxAcc", "TimeOfMaxAcc", "DIST"]]
        rows2['stationCHN'] = rows2['station']  + rows2['CHN']  
        rows2 = rows2.set_index('stationCHN')  
        rows2 = rows2.loc[rows2['CHN'].isin(['HNE','HNN','HNZ','BLE','BLN','BLZ'])][["station","CHN","MaxAcc", "TimeOfMaxAcc", "DIST"]]
        rows2['CHN'] = rows2['CHN'].str[-1:]
        rows2 = rows2.set_index(rows2['station']  + rows2['CHN'])  

        cols3 =["NET.STATION.LOCID.CMPNM", "MaxVel", "TimeOfMaxVel", "DIST", "AZ", "BAZ",
                "PeakVecVel", "EstimatedDuration."]
        df3 = pd.read_csv(self.path_summary_acc_text_file,names=cols3,sep=' ',skiprows=1)
        df3['station'] = df3['NET.STATION.LOCID.CMPNM'].str[3:7]
        df3['CHN'] = df3['NET.STATION.LOCID.CMPNM'].str[-3:]
        rows3 = df3.loc[df3['station'].isin(self.stations_name)][["station","CHN","MaxVel", "TimeOfMaxVel", "DIST"]]
        rows3['stationCHN'] = rows3['station']  + rows3['CHN']  
        rows3 = rows3.set_index('stationCHN')  
        rows3 = rows3.loc[rows3['CHN'].isin(['HNE','HNN','HNZ','BLE','BLN','BLZ'])][["station","CHN","MaxVel", "TimeOfMaxVel", "DIST"]]
        rows3['CHN'] = rows3['CHN'].str[-1:]
        rows3 = rows3.set_index(rows3['station']  + rows3['CHN'])  



        ### Construct dictionary to store data per station ###


        stations_data = dict()
        depth = self.earthquake_info[' Depth']
        i=0
        for station in self.stations_name:     
                
                stations_data[station]= dict()
                stations_data[station]['name']= station
                stations_data[station]['azimuth'] = float(rows.loc[station,'azi'][0])
                stations_data[station]['P-arrival time'] = UTCDateTime(datetime.strptime('{} '.format(self.date)+
                              rows.loc[station,'time'][0],"%Y-%m-%d %H:%M:%S.%f").isoformat())
                stations_data[station]['S-arrival time'] = UTCDateTime(datetime.strptime('{} '.format(self.date) +
                              rows.loc[station,'time'][1],"%Y-%m-%d %H:%M:%S.%f").isoformat())
                stations_data[station]['maxAccZ'] = float(rows2.loc[station+'Z','MaxAcc'])
                stations_data[station]['maxAccN'] = float(rows2.loc[station+'N','MaxAcc'])
                stations_data[station]['maxAccE'] = float(rows2.loc[station+'E','MaxAcc'])
                stations_data[station]['TimeOfmaxAccZ'] = UTCDateTime(datetime.strptime('{} '.format(self.date) +
                            rows2.loc[station+'Z','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
                stations_data[station]['TimeOfmaxAccN'] =  UTCDateTime(datetime.strptime('{} '.format(self.date) +
                            rows2.loc[station+'N','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
                stations_data[station]['TimeOfmaxAccE'] =  UTCDateTime(datetime.strptime('{} '.format(self.date) +
                            rows2.loc[station+'E','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
                stations_data[station]['maxVelZ'] = float(rows3.loc[station+'Z','MaxVel'])
                stations_data[station]['maxVelN'] = float(rows3.loc[station+'N','MaxVel'])
                stations_data[station]['maxVelE'] = float(rows3.loc[station+'E','MaxVel'])
                stations_data[station]['TimeOfmaxVelZ'] = UTCDateTime(datetime.strptime('{} '.format(self.date) +
                            rows3.loc[station+'Z','TimeOfMaxVel'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
                stations_data[station]['TimeOfmaxVelN'] =  UTCDateTime(datetime.strptime('{} '.format(self.date) +
                            rows3.loc[station+'N','TimeOfMaxVel'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
                stations_data[station]['TimeOfmaxVelE'] =  UTCDateTime(datetime.strptime('{} '.format(self.date) +
                            rows3.loc[station+'E','TimeOfMaxVel'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
                stations_data[station]['epicentralDist'] = float(rows2.loc[station+'E','DIST'])
                stations_data[station]['hypocentralDist'] = round(np.sqrt(float(rows2.loc[station+'E','DIST'])**2 + depth**2),2)
                stations_data[station]['absolute_path_N'] = Earthquake.url_finder(self.data_set_abs_path,station,'N')
                stations_data[station]['absolute_path_E'] = Earthquake.url_finder(self.data_set_abs_path,station,'E')
                stations_data[station]['absolute_path_Z'] = Earthquake.url_finder(self.data_set_abs_path,station,'Z')

                
                i=i+2
        self.stations_info = stations_data
        
        ### move this to end ###
        
        df_data = pd.DataFrame.from_dict(stations_data.values()) 
        df_data = df_data.set_index('name')
        df_data = df_data.sort_values(by=['hypocentralDist'])
        self.station_df = df_data
        path_stations_data = os.path.join(self.path_csv_files,'summary_stations_data.csv')
        df_data.to_csv(path_stations_data, header=True)
        
        
        
            
    @staticmethod
    def moving_ave(A, N):
        
        """
        Taken from Noisepy  from "https://github.com/mdenolle/NoisePy/blob/master/src/noisepy/seis/noise_module.py"
        
        Alternative function for moving average for an array.
        PARAMETERS:
        ---------------------
        A: 1-D array of data to be smoothed
        N: integer, it defines the full!! window length to smooth
        RETURNS:
        ---------------------
        B: 1-D array with smoothed data
        """
        # defines an array with N extra samples at either side
        temp = np.zeros(len(A) + 2 * N)
        # set the central portion of the array to A
        temp[N:-N] = A
        # leading samples: equal to first sample of actual array
        temp[0:N] = temp[N]
        # trailing samples: Equal to last sample of actual array
        temp[-N:] = temp[-N - 1]
        # convolve with a boxcar and normalize, and use only central portion of the result
        # with length equal to the original array, discarding the added leading and trailing samples
        B = np.convolve(temp, np.ones(N) / N, mode="same")[N:-N]
        return B
    
    @staticmethod    
    def raw_spectrum_subplot(df_station,f_max,scale_type,comp):  ### change from -3 secons before p-arrival and 3 seconds after coda wave lapse time
       
        n = len(df_station.index)
        fig,axes = plt.subplots(n,figsize=(12,15))
        abspath = 'absolute_path_{}'.format(comp)
        for i,station in enumerate(df_station.index):  
                trace = obs.read(df_station.loc[station,abspath])
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
                if scale_type == "linear":
                    axes[i].plot(freqs,power_spec,linewidth=0.5)
                    axes[i].set_xlim(0, f_max)
                    
                elif scale_type == 'log':
                    axes[i].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i].set_xlim(0.1, f_max)
                axes[i].set_ylabel('Power Spectrum ',fontsize=10)
                axes[i].set_xlabel('Frequency [Hz]',fontsize=10)
                axes[i].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i].tick_params(axis='both',labelsize=9)
                axes[i].set_title(station_name + ' ' + station_channel + '  Hypocentral Distance: {} km'.format(df_station.loc[station,'hypocentralDist']),fontdict={'fontsize': 12, 'fontweight':'bold','color':'blue'})
        return fig, axes
    
    @staticmethod
    def slicer(data_list,n_group=3):
        sl = []
        n_data = len(data_list)
        for i in range(0,n_data,n_group):
            if i==n_group*(n_data//n_group):
                sl.append((i,i+n_data%n_group))
            else:
                sl.append((i,i+n_group))
        return sl
    
    def raw_spectrum_plots(self,fmax,scaletype,plot_type='Raw_spectrum'):
        path_plot_type = os.path.join(self.path_plots,plot_type)
        if os.path.isdir(path_plot_type):
            shutil.rmtree(path_plot_type)
        os.makedirs(path_plot_type)
        
    ### Plotting spectra per each subset ###
        for comp in ['E','N','Z']:
            index_slice = Earthquake.slicer(self.station_df['absolute_path_{}'.format(comp)])
            for group in index_slice:
                fig,axes = Earthquake.raw_spectrum_subplot(self.station_df.iloc[group[0]:group[1]],fmax,scaletype,comp)   
                fig.suptitle(self.title +'\n {}'.format(plot_type.replace('_',' ')),fontsize=14)
                fig.tight_layout(pad=1.25)
                path_plot = os.path.join(path_plot_type,plot_type +"_comp_{}".format(comp)+"_{}_{}_".format(group[0],group[1])+"_fmax_{}_".format(fmax) + "_scale_{}".format(scaletype)+".jpeg")
                fig.savefig(path_plot,dpi=600)  
                

            
    @staticmethod    
    def noise_spectrum_subplot(df_station,f_max,scale_type,origin_time,comp,dt_bo=0,dt_ap=0):
        n = len(df_station.index) 
        fig,axes = plt.subplots(n,figsize=(12,15))
        abspath = 'absolute_path_{}'.format(comp)
        for i,station in enumerate(df_station.index):
                trace = obs.read(df_station.loc[station,abspath])
                trace.trim(origin_time+dt_bo,df_station.loc[station,'P-arrival time']+dt_ap)
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
                if scale_type == "linear":
                    axes[i].plot(freqs,power_spec,linewidth=0.5)
                    axes[i].set_xlim(0, f_max)
                elif scale_type == 'log':
                    axes[i].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i].set_xlim(0.1, f_max)
                axes[i].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i].tick_params(axis='both',labelsize=9)
                axes[i].set_title(station_name + ' ' + station_channel + '  Hypocentral Distance: {} km'.format(df_station.loc[station,'hypocentralDist']),fontdict={'fontsize': 12, 'fontweight':'bold','color':'blue'})        
                
        return fig, axes
    
    def noise_spectrum_plots(self,fmax,scaletype,dt_b,dt_a,plot_type='Noise_Spectrum',):
        path_plot_type = os.path.join(self.path_plots,plot_type)
        if os.path.isdir(path_plot_type):
            shutil.rmtree(path_plot_type)
        os.makedirs(path_plot_type)
       
    
        ### Plotting spectra per each subset ###
        t_origin = self.earthquake_info[' Origin Time']
        for comp in ['E','N','Z']:
            index_slice = Earthquake.slicer(self.station_df['absolute_path_{}'.format(comp)])       
            for group in index_slice:
                fig,axes = Earthquake.noise_spectrum_subplot(self.station_df.iloc[group[0]:group[1]],fmax,scaletype,t_origin,comp)   
                fig.suptitle(self.title +'\n {}'.format(plot_type.replace('_',' ')),fontsize=14)
                path_plot = os.path.join(path_plot_type,plot_type +"_comp_{}".format(comp)+"_{}_{}_".format(group[0],group[1])+"_fmax_{}_".format(fmax) + "_scale_{}".format(scaletype)+".jpeg")
                fig.tight_layout(pad=1.25)
                fig.savefig(path_plot,dpi=600)  
                
    @staticmethod
    def p_wave_spectrum_subplot(df_station,f_max,scale_type,comp,dt_adjL=0,dt_adjR=0):
        n = len(df_station.index)   
        abspath = 'absolute_path_{}'.format(comp)
        fig,axes = plt.subplots(n,figsize=(12,15))
        for i,station in enumerate(df_station.index):
                trace = obs.read(df_station.loc[station,abspath])
                trace.trim(df_station.loc[station,'P-arrival time']+dt_adjL,df_station.loc[station,'S-arrival time']+dt_adjR)
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
                if scale_type == "linear":
                    axes[i].plot(freqs,power_spec,linewidth=0.5)
                    axes[i].set_xlim(0, f_max)
                elif scale_type == 'log':
                    axes[i].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i].set_xlim(0.1, f_max)
                axes[i].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i].tick_params(axis='both',labelsize=9)
                axes[i].set_title(station_name + ' ' + station_channel + '  Hypocentral Distance: {} km'.format(df_station.loc[station,'hypocentralDist']),fontdict={'fontsize': 12, 'fontweight':'bold','color':'blue'})        
        return fig, axes
        
    def p_wave_spectrum_plots(self,fmax,scaletype,plot_type='P_wave_Spectrum',dt_adjustL = 0,dt_adjustR = 0):
        path_plot_type = os.path.join(self.path_plots,plot_type)
        if os.path.isdir(path_plot_type):
            shutil.rmtree(path_plot_type)
        os.mkdir(path_plot_type)    
    
        
        ### Plotting spectra per each subset ###
        for comp in ['E','N','Z']:
            index_slice = Earthquake.slicer(self.station_df['absolute_path_{}'.format(comp)])       
            for group in index_slice:
                fig,axes = Earthquake.p_wave_spectrum_subplot(self.station_df.iloc[group[0]:group[1]],fmax,scaletype,comp,dt_adjL=dt_adjustL,dt_adjR=dt_adjustR)                                   
                fig.suptitle(self.title +'\n {}'.format(plot_type.replace('_',' ')),fontsize=12)
                fig.tight_layout(pad=1.25)
                path_plot = os.path.join(path_plot_type,plot_type +"_comp_{}".format(comp)+"_{}_{}_".format(group[0],group[1])+"_fmax_{}_".format(fmax) + "_scale_{}".format(scaletype)+".jpeg")
                fig.savefig(path_plot,dpi=600)  

                  
         # csv_creator(path_event_text_file,'station_phase_arrivals.csv',lineStationPhase,line_end=lineStationMagn) 
         # csv_creator(path_event_text_file,'station_magnitudes.csv',lineStationMagn)    
     
    @staticmethod    
    def s_wave_spectrum_subplot(files_list,f_max,scale_type,dict_info,origin_time,dt_adj=0):
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(12,16))
        for i in range(0,len(files_list),3):
            for j in range(3):
                dt_s_wave = dict_info[files_list[i+j][3:7]]['S-arrival time'] - origin_time
                trace = obs.read('data/'+files_list[i+j])
                trace.trim(dict_info[files_list[i+j][3:7]]['S-arrival time'],dict_info[files_list[i+j][3:7]]['S-arrival time']+ dt_s_wave + dt_adj)  # s-wave windows lasts 2 times its arrival time
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
                if scale_type == "linear":
                    axes[i//3][j].plot(freqs,power_spec,linewidth=0.5)
                    axes[i//3][j].set_xlim(0, f_max)
                    
                elif scale_type == 'log':
                    axes[i//3][j].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i//3][j].set_xlim(0.1, f_max)
                axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i//3][j].tick_params(axis='both',labelsize=7.5)
                axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 9,'color':'blue','fontweight':'bold'})
        return fig, axes
    
    
    def s_wave_spectrum_plots(self,fmax,scaletype,plot_type='S_wave_Spectrum',dt_adjust = 0):
        path_plot_type = os.path.join(self.path_plots,plot_type)
        if os.path.isdir(path_plot_type):
            shutil.rmtree(path_plot_type)
        os.mkdir(path_plot_type)    
         
         
         ### Plotting spectra per each subset ###
        t_origin = self.earthquake_info[' Origin Time']
        index_slice = Earthquake.slicer(self.data_set)
        for group in index_slice:
            fig,axes = Earthquake.s_wave_spectrum_subplot(self.data_set[group[0]:group[1]],fmax,scaletype,self.stations_info,t_origin,dt_adj = dt_adjust)                                  
            fig.suptitle(self.title +'\n {}'.format(plot_type.replace('_',' ')),fontsize=12)
            fig.supxlabel(r'f')
            fig.supylabel(r'Power Spectrum')
            fig.tight_layout(pad=1.25)
            path_plot = os.path.join(path_plot_type,plot_type +"_{}_{}".format(group[0],group[1])+"_fmax_{}_".format(fmax) + "_scale_{}".format(scaletype)+".jpeg")
            fig.savefig(path_plot,dpi=600)  

    
    @staticmethod        
    def coda_wave_spectrum_subplot(files_list,f_max,scale_type,dict_info,origin_time,factor_start = 2,factor_end = 4):
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(12,16))
        for i in range(0,len(files_list),3):
            for j in range(3):
                dt_s_wave = dict_info[files_list[i+j][3:7]]['S-arrival time'] - origin_time
                trace = obs.read('data/'+files_list[i+j])
                trace.trim(origin_time+factor_start*dt_s_wave,origin_time+factor_end*dt_s_wave)  # s-wave windows lasts 2 times its arrival time
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
                if scale_type == "linear":
                    axes[i//3][j].plot(freqs,power_spec,linewidth=0.5)
                    axes[i//3][j].set_xlim(0, f_max)
                elif scale_type == 'log':
                    axes[i//3][j].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i//3][j].set_xlim(0.1, f_max)
                axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                axes[i//3][j].tick_params(axis='both',labelsize=6)
                axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
                   
        return fig, axes
    def coda_wave_spectrum_plots(self,fmax,scaletype,plot_type='Coda_wave_Spectrum',dt_adjust = 0,factor_s = 2, factor_e = 4):
        path_plot_type = os.path.join(self.path_plots,plot_type)
        if os.path.isdir(path_plot_type):
            shutil.rmtree(path_plot_type)
        os.mkdir(path_plot_type)    
        ### Plotting spectra per each subset ###
        t_origin = self.earthquake_info[' Origin Time']
        index_slice = Earthquake.slicer(self.data_set)
        for group in index_slice:
            fig,axes = Earthquake.coda_wave_spectrum_subplot(self.data_set[group[0]:group[1]],fmax,scaletype,self.stations_info,t_origin,factor_start=factor_s,factor_end = factor_e)                                    
            fig.suptitle(self.title +'\n {}'.format(plot_type.replace('_',' ')),fontsize=12)
            fig.supxlabel(r'f')
            fig.supylabel(r'Power Spectrum')
            fig.tight_layout(pad=1.25)
            path_plot = os.path.join(path_plot_type,plot_type +"_{}_{}".format(group[0],group[1])+"_fmax_{}_".format(fmax) + "_scale_{}".format(scaletype)+".jpeg")
            fig.savefig(path_plot,dpi=600)  

    

    @staticmethod
    def intensity_simps(y,x,i):
        return simpson(y[:i],x[:i])/simpson(y,x)   
    
    @staticmethod
    def intensity_subplot(files_list,t_max,origin_time,dict_info,comp):
        components = {'E':0,'N':1,'Z':2}    
        j = components[comp]        
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,figsize=(12,16))
        
        # fig1,axes1 = plt.subplots(n,3,figsize=(12,16))
        for i in range(0,len(files_list),3):

            g = 980.67
            trace = obs.read('data/'+files_list[i+j])
            ending_time = trace[0].stats.endtime
            trace.trim(origin_time,ending_time)  # trace trimmed from event start to ending time
            tr = obs.read('data/'+files_list[i+j])
            data = trace[0]
            detrended_data = detrend(data,type='linear')
            times = data.times()
            station_name  = data.stats.station
            station_channel = data.stats.channel
            I_simps = [Earthquake.intensity_simps(np.array(detrended_data)**2,times,i) for i in range(1,len(times))]   
            AI = (np.pi/(2*g))*(simpson(np.array(detrended_data)**2,times))
            time = times[1:]
            init_percent = 0.05*np.ones(len(I_simps)) 
            end_percent = 0.95*np.ones(len(I_simps))
            idx_5 = np.argwhere(np.diff(np.sign(np.array(I_simps)-init_percent))).flatten()
            idx_95 = np.argwhere(np.diff(np.sign(np.array(I_simps)-end_percent))).flatten()
            x_eff_5 = (time[idx_5][0],np.array(I_simps)[idx_5][0])
            x_eff_95 = (time[idx_95][0],np.array(I_simps)[idx_95][0])
            ## Plotting Ground-Motion
            p_time = dict_info[files_list[i+j][3:7]]['P-arrival time']-origin_time
            s_time = dict_info[files_list[i+j][3:7]]['S-arrival time']-origin_time
            coda_time = 2*s_time
            timeMaxPGA = dict_info[files_list[i+j][3:7]]['TimeOfmaxAccZ']
            dtMaxPGA =   timeMaxPGA - origin_time
            maxPGA = tr.trim(timeMaxPGA,timeMaxPGA)[0][0]
            effD =time[idx_95][0] - time[idx_5][0]
            hyp_dist = dict_info[files_list[i+j][3:7]]['hypocentralDist']
            ## Plotting Arias intensity ##   
            axes1 = axes[i//3].twinx()
            axes[i//3].plot(times,data,'r-',linewidth=0.7)
            axes[i//3].plot(dtMaxPGA,maxPGA, color='green',marker='*',markersize=8)
            axes[i//3].tick_params(axis='both', labelsize=11)
            axes[i//3].set_xlim(0,t_max)
            axes[i//3].set_ylabel('Acceleration '+'[cm/'+r'$s^{2}$'+']',fontsize=10)
            axes[i//3].set_xlabel('Time [s]',fontsize=10)
            axes[i//3].axvline(p_time,color='c',linewidth=0.7)
            axes[i//3].axvline(s_time,color='c',linewidth=0.7)
            axes[i//3].axvline(coda_time,color='c',linewidth=0.7)
            axes[i//3].annotate(r'$I_{A} =$'+'{:.3f}'.format(AI/100) +' m/s', xy=(0.85*100, 0.92*max(data)), ha='center', va='top',fontsize=12)
            axes[i//3].annotate(r'$D_{5-95} =$'+'{:.2f}'.format(effD) +' s', xy=(0.85*100, 0.75*max(data)), ha='center', va='top',fontsize=12)
            axes[i//3].annotate(r'$∣PGA∣ =$'+'{:.2f}'.format(abs(maxPGA)) +' cm/'+r'$s^{2}$', xy=(0.85*100, 0.58*max(data)), ha='center', va='top',fontsize=12)
            axes[i//3].annotate(r'$D_{5-95}$', xy=(0.80*time[idx_95][0], max(data)/2.4), ha='center', va='top',fontsize=10)
            axes[i//3].annotate( "", xy=(time[idx_5][0],max(data)/2), xytext=(time[idx_95][0], max(data)/2),arrowprops=dict(arrowstyle="<|-|>,head_width=0.1",facecolor='k', linewidth=0.6, shrinkA=0, shrinkB=0) )
            axes[i//3].annotate('P\n', xy=(p_time,max(data)+0.08*max(data)), ha='center', va='top',fontsize=8,rotation=90)
            axes[i//3].annotate('S\n', xy=(s_time,max(data)+0.08*max(data)), ha='center', va='top',fontsize=8,rotation=90)
            axes[i//3].annotate('Coda\n', xy=(coda_time,max(data)+0.08*max(data)), ha='center', va='top',fontsize=8,rotation = 90)
            axes[i//3].axvline(time[idx_5][0],color='b',linestyle='--',linewidth=0.7)
            axes[i//3].axvline(time[idx_95][0],color='b',linestyle='--',linewidth=0.7)
            axes[i//3].set_title(station_name + ' ' + station_channel +'   ' + 'Hypocentral Distance: {} km'.format(hyp_dist),fontdict={'fontsize': 12, 'fontweight':'bold','color':'blue'})
           
            axes1.set_ylabel('Norm.$I_{A}(t)$' ,fontsize=10)
            axes1.plot(time,I_simps,color='tab:orange',linewidth=1.2)
            axes1.plot(x_eff_5[0],x_eff_5[1],'yo',x_eff_95[0],x_eff_95[1],'yo',markersize=5)
            axes1.set_xlim(0,t_max)
            axes1.tick_params(axis='both', labelsize=11)
            axes1.set_xticks(np.arange(0,t_max+1,20))
                # axes1.axhline(0.05,color='g',linewidth=0.4)
                # axes1.axhline(0.95,color='g',linewidth=0.4)
                #axes1.set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
               
                
        
        
        return fig,axes,#fig1,axes1
    def intensity_plots(self,tmax,plot_type='Acceleration_and_Arias_Intensity'):
        path_plot_type = os.path.join(self.path_plots,plot_type)
        if os.path.isdir(path_plot_type):
            shutil.rmtree(path_plot_type)
        os.mkdir(path_plot_type)  
          
        t_origin = self.earthquake_info[' Origin Time']
        index_slice = Earthquake.slicer(self.data_set)
        for comp in ['E','N','Z']:
            for group in index_slice:
            ### Plotting spectra per each subset ###
                fig,axes= Earthquake.intensity_subplot(self.data_set[group[0]:group[1]],tmax,t_origin,self.stations_info,comp)                                   
                fig.suptitle(self.title +'\n {}'.format(plot_type.replace('_',' ')),fontsize=12)
                # fig.supylabel(r'Intensity')
                # fig.supxlabel(r'Time')
                fig.tight_layout(pad=1.25)
                path_plot = os.path.join(path_plot_type,plot_type +"_{}_{}".format(group[0],group[1])+"_comp_{}".format(comp)+"_t_max_{}_".format(tmax) +".jpeg")
                fig.savefig(path_plot,dpi=600)  
            
#     """
    
#     data_folder_path: absolute path to SAC files folder
#     wave_picking_file: csv file with info on
#     stations_file: csv file with info stations (distance, ) 
    
#     """
#     ### start from path to data folder where SAC files are located  
    
#     data_file_list = os.listdir(data_folder_path)
    
#     ### setting stations name ###

#     stations_name = list({file[3:7] for file in data_files_list}
           
#     for statio in station_name                     
#     trace = obs.read(files_list[j])
    
    
    
# main_dir = os.path.abspath(os.getcwd()) 
# data_folder_dir =  os.path.join(main_dir,"data")
# data_files_list = os.listdir(data_folder_dir)
# stations_name =  list({file[3:7] for file in data_files_list})
# data_files_list = [os.path.join(data_folder_dir,item) for item in os.listdir(data_folder_dir)]


# def line_index(path:str,string:str):
#     with open(path, 'r') as fp:
#         for l_no, line in enumerate(fp):
#             # search string
#             if string in line:
#                 c =l_no
#                 break
#                 # don't look for next lines
#     return c
    
# def string_finder(path:str,string:str):
#     with open(path, 'r') as fp:
#         for l_no, line in enumerate(fp):
#             # search string
#             if string in line: 
#                 c = line
#                 break
#                 # don't look for next lines
#     return c
        

# ### event id ###
# event_id = string_finder(path_event_text_file,'Public ID')
# event_id_regex = re.compile(r'(\w{5}\d{4}\w{4})')
# event_id = re.findall(event_id_regex,event_id)[0]

# ### event date and time ###
# date = string_finder(path_event_text_file,'Date') 
# date_regex = re.compile(r'(\d{4}-\d{2}-\d{2})')
# date = re.findall(date_regex,date)[0]

# time = string_finder(path_event_text_file,'Time') 
# time_regex = re.compile(r'(\d{2}:\d{2}:\d*\.?\d*)')
# time = re.findall(time_regex,time)[0]

# event_origin = datetime.strptime(date +' '+time, "%Y-%m-%d %H:%M:%S.%f")
# t0 = event_origin.isoformat()                     #ISO format
# t0 = UTCDateTime(t0)  

#  ### event latitude,  and depth ### 
 
# latitude = string_finder(path_event_text_file,'Latitude') 
# latitude_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
# latitude = re.findall(latitude_regex,latitude)[0]
# latitude = float(latitude)
   
# longitude = string_finder(path_event_text_file,'Longitude') 
# longitude_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
# longitude = re.findall(longitude_regex,longitude)[0]
# longitude = float(longitude)

# depth = string_finder(path_event_text_file,'Depth') 
# depth_regex = re.compile(r'[-+]?(?:\d*\.\d+|\d+)')
# depth = re.findall(depth_regex,depth)[0]
# depth = float(depth)


# ### General information on the earthquake considered ###
# earthquake_info = {'event_id':event_id,'origin_time':t0,'latitude':latitude,'longitude':longitude,'depth':depth}




# lineStationPhase = line_index(path_event_text_file,'Phase arrivals:')  
# lineStationMagn = line_index(path_event_text_file,'Station magnitudes:')


# def csv_creator(path:str,name_csv:str,line_start:int,line_end=None):
#     """
    

#     Parameters
#     ----------
#     path : str
#         Path to text file containing event generated by Seiscom
#     name_csv : str
#         Name of csv file 
#     line_start : int
#         Line to start at
#     line_end : TYPE, optional
#         Line to end at

#     Returns
#     -------
#     None.

#     """
#     file = open(path)
#     content = file.readlines()[line_start:line_end]
#     content = [re.sub(r'\s+',",",line) for line in content]
#     content = [re.findall('(?<=,)[^,]+(?=,)',item)[:6] for item in content][1:]   
#     csv_file = open('{}'.format(name_csv), 'w+', newline ='')
#     with csv_file:   
#         write = csv.writer(csv_file)
#         write.writerows(content)
#     return content
         
# csv_creator(path_event_text_file,'station_phase_arrivals.csv',lineStationPhase,line_end=lineStationMagn) 
# csv_creator(path_event_text_file,'station_magnitudes.csv',lineStationMagn)    

# ### Filter data to keep only desired stations . Create data frames ### 


# cols = ['sta','net','dist','azi','phase','time']
# df = pd.read_csv("station_phase_arrivals.csv",names=cols,sep=',',skiprows=1)
# rows = df.loc[df['sta'].isin(stations_name)][cols]
# rows = rows.set_index('sta')

# cols1 = ['sta','net','dist','azi','type','value']
# df1 = pd.read_csv("station_magnitudes.csv",names=cols1,sep=',',skiprows=1)
# rows1 = df1.loc[df1['sta'].isin(stations_name)][cols1]
# rows1 = rows1.set_index('sta')

# cols2 =["NET.STATION.LOCID.CMPNM", "MaxAcc", "TimeOfMaxAcc", "DIST", "AZ", "BAZ",
#         "PeakVecACC", "EstimatedDuration."]
# df2 = pd.read_csv(path_summary_acc_text_file,names=cols2,sep=' ',skiprows=1)
# df2['station'] = df2['NET.STATION.LOCID.CMPNM'].str[3:7]
# df2['CHN'] = df2['NET.STATION.LOCID.CMPNM'].str[-3:]
# rows2 = df2.loc[df2['station'].isin(stations_name)][["station","CHN","MaxAcc", "TimeOfMaxAcc", "DIST"]]
# rows2['stationCHN'] = rows2['station']  + rows2['CHN']  
# rows2 = rows2.set_index('stationCHN')  
# rows2 = rows2.loc[rows2['CHN'].isin(['HNE','HNN','HNZ','BLE','BLN','BLZ'])][["station","CHN","MaxAcc", "TimeOfMaxAcc", "DIST"]]
# rows2['CHN'] = rows2['CHN'].str[-1:]
# rows2 = rows2.set_index(rows2['station']  + rows2['CHN'])  

# cols3 =["NET.STATION.LOCID.CMPNM", "MaxVel", "TimeOfMaxVel", "DIST", "AZ", "BAZ",
#         "PeakVecVel", "EstimatedDuration."]
# df3 = pd.read_csv(path_summary_acc_text_file,names=cols3,sep=' ',skiprows=1)
# df3['station'] = df3['NET.STATION.LOCID.CMPNM'].str[3:7]
# df3['CHN'] = df3['NET.STATION.LOCID.CMPNM'].str[-3:]
# rows3 = df3.loc[df3['station'].isin(stations_name)][["station","CHN","MaxVel", "TimeOfMaxVel", "DIST"]]
# rows3['stationCHN'] = rows3['station']  + rows3['CHN']  
# rows3 = rows3.set_index('stationCHN')  
# rows3 = rows3.loc[rows3['CHN'].isin(['HNE','HNN','HNZ','BLE','BLN','BLZ'])][["station","CHN","MaxVel", "TimeOfMaxVel", "DIST"]]
# rows3['CHN'] = rows3['CHN'].str[-1:]
# rows3 = rows3.set_index(rows3['station']  + rows3['CHN'])  



# ### Construct dictionary to store data per station ###


# stations_data = dict()

# for station in stations_name:
#         stations_data[station]= dict()
#         stations_data[station]['name']= station
#         stations_data[station]['azimuth'] = float(rows.loc[station,'azi'][0])
#         stations_data[station]['P-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+
#                      rows.loc[station,'time'][0],"%Y-%m-%d %H:%M:%S.%f").isoformat())
#         stations_data[station]['S-arrival time'] = UTCDateTime(datetime.strptime('2023-03-18 '+
#                      rows.loc[station,'time'][1],"%Y-%m-%d %H:%M:%S.%f").isoformat())
#         stations_data[station]['maxAccZ'] = float(rows2.loc[station+'Z','MaxAcc'])
#         stations_data[station]['maxAccN'] = float(rows2.loc[station+'N','MaxAcc'])
#         stations_data[station]['maxAccE'] = float(rows2.loc[station+'E','MaxAcc'])
#         stations_data[station]['TimeOfmaxAccZ'] = UTCDateTime(datetime.strptime('2023-03-18 '+
#                     rows2.loc[station+'Z','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
#         stations_data[station]['TimeOfmaxAccN'] =  UTCDateTime(datetime.strptime('2023-03-18 '+
#                     rows2.loc[station+'N','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
#         stations_data[station]['TimeOfmaxAccE'] =  UTCDateTime(datetime.strptime('2023-03-18 '+
#                     rows2.loc[station+'E','TimeOfMaxAcc'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
#         stations_data[station]['maxVelZ'] = float(rows3.loc[station+'Z','MaxVel'])
#         stations_data[station]['maxVelN'] = float(rows3.loc[station+'N','MaxVel'])
#         stations_data[station]['maxVelE'] = float(rows3.loc[station+'E','MaxVel'])
#         stations_data[station]['TimeOfmaxVelZ'] = UTCDateTime(datetime.strptime('2023-03-18 '+
#                     rows3.loc[station+'Z','TimeOfMaxVel'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
#         stations_data[station]['TimeOfmaxVelN'] =  UTCDateTime(datetime.strptime('2023-03-18 '+
#                     rows3.loc[station+'N','TimeOfMaxVel'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
#         stations_data[station]['TimeOfmaxVelE'] =  UTCDateTime(datetime.strptime('2023-03-18 '+
#                     rows3.loc[station+'E','TimeOfMaxVel'],"%Y-%m-%d %H:%M:%S.%f").isoformat())
#         stations_data[station]['epicentralDist'] = float(rows2.loc[station+'E','DIST'])
#         stations_data[station]['hypocentralDist'] = round(np.sqrt(float(rows2.loc[station+'E','DIST'])**2 + depth**2),2)
    
 
    