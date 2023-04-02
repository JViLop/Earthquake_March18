
import numpy as np
from scipy.signal import find_peaks,detrend
import obspy as obs
import os
import shutil
from scipy.integrate import trapezoid,simpson
import matplotlib.pylab as plt


""" 

        RAW SPECTRUM 
        
        For each station, raw spectrum is obtained; processing includes 
        only detrending


"""
def raw_spectrum_plots(main_dir,files_set,xlim,scale,t0,plot_type='Raw_Spectrum'):
    plots_dir = os.path.join(main_dir,"plots")
    plot_dir = os.path.join(plots_dir,plot_type)
    file_dir = os.path.join(plot_dir,plot_type +"_" + scale +"_fmax_"+ str(xlim) + ".jpeg")
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)
    
    def raw_spectrum_subplot(files_list,x_lim,scale_type):  ### change from -3 secons before p-arrival and 3 seconds after coda wave lapse time
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(10,14))
        for i in range(0,len(files_list),3):
            for j in range(3):
                trace = obs.read('data/'+files_list[i+j])
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
                    axes[i//3][j].set_xlim(0, x_lim)
                    axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                    axes[i//3][j].tick_params(axis='both',labelsize=6)
                    axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
                elif scale_type == 'log':
                    axes[i//3][j].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i//3][j].set_xlim(0.1, x_lim)
                    axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                    axes[i//3][j].tick_params(axis='both',labelsize=6)
                    axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
        return fig, axes

### Plotting spectra per each subset ###

    fig,axes = raw_spectrum_subplot(files_set,xlim,scale)                                  
    fig.suptitle('Event: igepn2023fkei Time: {}'.format(t0),fontsize=8)
    fig.supxlabel(r'f')
    fig.supylabel(r'Power Spectrum')
    fig.tight_layout(pad=1.25)
    fig.savefig(file_dir,dpi=600)  
    
    
    
""" 
        NOISE SPECTRUM 
        
"""    

def noise_spectrum_plots(main_dir,files_set,xlim,scale,stations_data,t0,dt_b,dt_a,plot_type='Noise_Spectrum',):
    plots_dir = os.path.join(main_dir,"plots")
    plot_dir = os.path.join(plots_dir,plot_type)
    file_dir = os.path.join(plot_dir,plot_type +"_" + scale +"_fmax_"+ str(xlim) + ".jpeg")
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)
    
    def noise_spectrum_subplot(files_list,x_lim,scale_type,origin_time,dict_info,dt_bo=0,dt_ap=0):
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(10,14))
        for i in range(0,len(files_list),3):
            for j in range(3):
                trace = obs.read('data/'+files_list[i+j])
                trace.trim(origin_time+dt_b,dict_info[files_list[i+j][3:7]]['P-arrival time']+dt_a)
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
                    axes[i//3][j].set_xlim(0, x_lim)
                    axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                    axes[i//3][j].tick_params(axis='both',labelsize=6)
                    axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
                elif scale_type == 'log':
                    axes[i//3][j].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i//3][j].set_xlim(0.1, x_lim)
                    axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                    axes[i//3][j].tick_params(axis='both',labelsize=6)
                    axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
        return fig, axes

    ### Plotting spectra per each subset ###

    fig,axes = noise_spectrum_subplot(files_set,xlim,scale,t0,stations_data,dt_bo = dt_b,dt_ap = dt_a)                                
    fig.suptitle('Event: igepn2023fkei Time: {} \n Noise prior to P-wave arrival'.format(t0),fontsize=10)
    fig.supxlabel(r'f')
    fig.supylabel(r'Power Spectrum')
    fig.tight_layout(pad=1.25)
    fig.savefig(file_dir,dpi=600)  

""" 

        P-WAVE SPECTRUM 
        
"""    
    
def p_wave_spectrum_plots(main_dir,files_set,xlim,scale,stations_data,t0,plot_type='P_wave_Spectrum',dt_adjust = 0):
    plots_dir = os.path.join(main_dir,"plots")
    plot_dir = os.path.join(plots_dir,plot_type)
    file_dir = os.path.join(plot_dir,plot_type +"_" + scale +"_fmax_"+ str(xlim) + ".jpeg")
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)    

    def p_wave_spectrum_subplot(files_list,x_lim,scale_type,dict_info,dt_adj=0):
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(10,14))
        for i in range(0,len(files_list),3):
            for j in range(3):
                    trace = obs.read('data/'+files_list[i+j])
                    trace.trim(dict_info[files_list[i+j][3:7]]['P-arrival time']+dt_adj,dict_info[files_list[i+j][3:7]]['S-arrival time']+dt_adj)
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
                        axes[i//3][j].set_xlim(0, x_lim)
                        axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                        axes[i//3][j].tick_params(axis='both',labelsize=6)
                        axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
                    elif scale_type == 'log':
                        axes[i//3][j].semilogx(freqs,power_spec,linewidth=0.5)
                        axes[i//3][j].set_xlim(0.1, x_lim)
                        axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                        axes[i//3][j].tick_params(axis='both',labelsize=6)
                        axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
        return fig, axes
    
    ### Plotting spectra per each subset ###
    
    fig,axes = p_wave_spectrum_subplot(files_set,xlim,scale,stations_data,dt_adj=dt_adjust)                                   
    fig.suptitle('Event: igepn2023fkei Time: {} \n P-wave lapse time'.format(t0))
    fig.supxlabel(r'f')
    fig.supylabel(r'Power Spectrum')
    fig.tight_layout(pad=1.25)
    fig.savefig(file_dir,dpi=600)  

""" 

        S-WAVE SPECTRUM 
        
"""    

def s_wave_spectrum_plots(main_dir,files_set,xlim,scale,stations_data,t0,plot_type='S_wave_Spectrum',dt_adjust = 0):
    plots_dir = os.path.join(main_dir,"plots")
    plot_dir = os.path.join(plots_dir,plot_type)
    file_dir = os.path.join(plot_dir,plot_type +"_" + scale +"_fmax_"+ str(xlim) + ".jpeg")
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)    
    

    def s_wave_spectrum_subplot(files_list,x_lim,scale_type,dict_info,origin_time,dt_adj=0):
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(10,14))
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
                    axes[i//3][j].set_xlim(0, x_lim)
                    axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                    axes[i//3][j].tick_params(axis='both',labelsize=6)
                    axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
                elif scale_type == 'log':
                    axes[i//3][j].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i//3][j].set_xlim(0.1, x_lim)
                    axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                    axes[i//3][j].tick_params(axis='both',labelsize=6)
                    axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
        return fig, axes
    
    ### Plotting spectra per each subset ###
    
    fig,axes = s_wave_spectrum_subplot(files_set,xlim,scale,stations_data,t0,dt_adj = dt_adjust)                                  
    fig.suptitle('Event: igepn2023fkei Time: {} \n S-wave lapse time'.format(t0))
    fig.supylabel(r'Power Spectrum')
    fig.supxlabel(r'f')
    fig.tight_layout(pad=1.25)
    fig.savefig(file_dir,dpi=600) 
    
"""

CODA-WAVE SPECTRUM

"""
def coda_wave_spectrum_plots(main_dir,files_set,xlim,scale,stations_data,t0,plot_type='Coda_wave_Spectrum',dt_adjust = 0,factor_s = 2, factor_e = 4):
    plots_dir = os.path.join(main_dir,"plots")
    plot_dir = os.path.join(plots_dir,plot_type)
    file_dir = os.path.join(plot_dir,plot_type +"_" + scale +"_fmax_"+ str(xlim) + ".jpeg")
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)    
    
    def coda_wave_spectrum_subplot(files_list,origin_time,x_lim,scale_type,dict_info,factor_start = 2,factor_end = 4):
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(10,14))
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
                    axes[i//3][j].set_xlim(0, x_lim)
                    axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                    axes[i//3][j].tick_params(axis='both',labelsize=6)
                    axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
                elif scale_type == 'log':
                    axes[i//3][j].semilogx(freqs,power_spec,linewidth=0.5)
                    axes[i//3][j].set_xlim(0.1, x_lim)
                    axes[i//3][j].ticklabel_format(style='sci',axis='y',scilimits=(0,0))
                    axes[i//3][j].tick_params(axis='both',labelsize=6)
                    axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
                   
        return fig, axes
    
    ### Plotting spectra per each subset ###
    
    fig,axes = coda_wave_spectrum_subplot(files_set,t0,xlim,scale,stations_data,factor_start=factor_s,factor_end = factor_e)                                    
    fig.suptitle('Event: igepn2023fkei Time: {} \n Coda wave lapse time'.format(t0))
    fig.supylabel(r'Power Spectrum')
    fig.supxlabel(r'f')
    fig.tight_layout(pad=1.25)
    fig.savefig(file_dir,dpi=600) 


""" 

EFFECTIVE DURATION in terms of INTENSITY

"""

def intensity_trapz(y,x,i):
    return trapezoid(y[:i],x[:i])/trapezoid(y,x)

def intensity_simps(y,x,i):
    return simpson(y[:i],x[:i])/simpson(y,x)   

def intensity_plots(main_dir,files_set,stations_data,t0,plot_type='Arias_Intensity'):
    plots_dir = os.path.join(main_dir,"plots")
    plot_dir = os.path.join(plots_dir,plot_type)
    file_dir = os.path.join(plot_dir,plot_type + ".jpeg")
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)    
      
    def intensity_subplot(files_list,origin_time,dict_info):
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,3,figsize=(12,15))
        for i in range(0,len(files_list),3):
            for j in range(3):
                trace = obs.read('data/'+files_list[i+j])
                ending_time = trace[0].stats.endtime
                trace.trim(origin_time,ending_time)  # trace trimmed from event start to ending time
                data = trace[0]
                detrended_data = detrend(data,type='linear')
                times = data.times()
                station_name  = data.stats.station
                station_channel = data.stats.channel
                I_simps = [intensity_simps(np.array(detrended_data)**2,times,i) for i in range(1,len(times))]   
                time = times[1:]
                init_percent = 0.05*np.ones(len(I_simps)) 
                end_percent = 0.95*np.ones(len(I_simps))
                idx_5 = np.argwhere(np.diff(np.sign(np.array(I_simps)-init_percent))).flatten()
                idx_95 = np.argwhere(np.diff(np.sign(np.array(I_simps)-end_percent))).flatten()
                x_eff_5 = (time[idx_5][0],np.array(I_simps)[idx_5][0])
                x_eff_95 = (time[idx_95][0],np.array(I_simps)[idx_95][0])
                ## Plotting ##    
                axes[i//3][j].plot(time,I_simps,'b-',linewidth=0.5)
                axes[i//3][j].plot(x_eff_5[0],x_eff_5[1],'ro',x_eff_95[0],x_eff_95[1],'ko',)
                axes[i//3][j].set_xlim(0,120)
                axes[i//3][j].set_xticks(np.arange(0,120,20))
                axes[i//3][j].axhline(0.05,color='r',linewidth=0.4)
                axes[i//3][j].axhline(0.95,color='r',linewidth=0.4)
                axes[i//3][j].set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
        return fig, axes
    
    ### Plotting spectra per each subset ###
    
    fig,axes = intensity_subplot(files_set,t0,stations_data)                                   
    fig.suptitle('Event: igepn2023fkei Time: {} \n Arias Intensity from event origin'.format(t0))
    fig.supylabel(r'Intensity')
    fig.tight_layout(pad=1.25)
    fig.savefig(file_dir,dpi=600)  
        