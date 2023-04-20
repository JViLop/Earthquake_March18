
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

def moving_ave(A, N):
    """
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
        fig,axes = plt.subplots(n,3,figsize=(12,16))
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
        fig,axes = plt.subplots(n,3,figsize=(12,16))
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
        fig,axes = plt.subplots(n,3,figsize=(12,16))
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


""" 

EFFECTIVE DURATION in terms of INTENSITY

"""
g = 980.67
def intensity_trapz(y,x,i):
    return trapezoid(y[:i],x[:i])/trapezoid(y,x)

def intensity_simps(y,x,i):
    return simpson(y[:i],x[:i])/simpson(y,x)   

def intensity_plots(main_dir,files_set,stations_data,t0,plot_type='Arias_Intensity_acceleration',componente='N'):
    plots_dir = os.path.join(main_dir,"plots")
    plot_dir = os.path.join(plots_dir,plot_type)
    file_dir= os.path.join(plot_dir,plot_type + ".jpeg")
    file_dir_acc= os.path.join(plot_dir,"Accelerations" + ".jpeg")
    if os.path.isdir(plot_dir):
        shutil.rmtree(plot_dir)
    os.mkdir(plot_dir)    
    
      
    def intensity_subplot(files_list,origin_time,dict_info,comp):
        if comp=='E':
            j=0
        elif comp=='N':
            j=1
        else:
            j=2
            
        n = len(files_list)//3    
        fig,axes = plt.subplots(n,figsize=(12,15))
        
        # fig1,axes1 = plt.subplots(n,3,figsize=(12,16))
        for i in range(0,len(files_list),3):

            trace = obs.read('data_ElOro/'+files_list[i+j])
            ending_time = trace[0].stats.endtime
            trace.trim(origin_time,ending_time)  # trace trimmed from event start to ending time
            tr = obs.read('data_ElOro/'+files_list[i+j])
            data = trace[0]
            detrended_data = detrend(data,type='linear')
            times = data.times()
            station_name  = data.stats.station
            station_channel = data.stats.channel
            I_simps = [intensity_simps(np.array(detrended_data)**2,times,i) for i in range(1,len(times))]   
            AI = (np.pi/(2*g))*(simpson(np.array(detrended_data)**2,times))
            time = times[1:]
            init_percent = 0.05*np.ones(len(I_simps)) 
            end_percent = 0.95*np.ones(len(I_simps))
            idx_5 = np.argwhere(np.diff(np.sign(np.array(I_simps)-init_percent))).flatten()
            idx_95 = np.argwhere(np.diff(np.sign(np.array(I_simps)-end_percent))).flatten()
            x_eff_5 = (time[idx_5][0],np.array(I_simps)[idx_5][0])
            x_eff_95 = (time[idx_95][0],np.array(I_simps)[idx_95][0])
            ## Plotting Ground-Motion
            p_time = dict_info[files_list[i+j][4:8]]['P-arrival time']-origin_time
            s_time = dict_info[files_list[i+j][4:8]]['S-arrival time']-origin_time
            coda_time = 2*s_time
            timeMaxPGA = dict_info[files_list[i+j][4:8]]['TimeOfmaxAccZ']
            dtMaxPGA =   timeMaxPGA - origin_time
            maxPGA = tr.trim(timeMaxPGA,timeMaxPGA)[0][0]
            effD =time[idx_95][0] - time[idx_5][0]
            hyp_dist = dict_info[files_list[i+j][4:8]]['hypocentralDist']
            ## Plotting Arias intensity ##   
            axes1 = axes[i//3].twinx()
            axes[i//3].plot(times,data,'r-',linewidth=0.7)
            axes[i//3].plot(dtMaxPGA,maxPGA, color='green',marker='*',markersize=8)
            axes[i//3].tick_params(axis='both', labelsize=11)
            axes[i//3].set_xlim(0,100)
            axes[i//3].set_ylabel('Aceleración '+'[cm/'+r'$s^{2}$'+']',fontsize=10)
            axes[i//3].set_xlabel('Tiempo [s]',fontsize=10)
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
            axes[i//3].set_title(station_name + ' ' + station_channel +'   ' + 'Distancia Hipocentral: '+str(hyp_dist)+ ' km',fontdict={'fontsize': 12, 'fontweight':'bold','color':'blue'})
           
            axes1.set_ylabel('$I_{A}(t)$ Normalizada' ,fontsize=10)
            axes1.plot(time,I_simps,color='tab:orange',linewidth=1.2)
            axes1.plot(x_eff_5[0],x_eff_5[1],'yo',x_eff_95[0],x_eff_95[1],'yo',markersize=5)
            axes1.set_xlim(0,100)
            axes1.tick_params(axis='both', labelsize=11)
            axes1.set_xticks(np.arange(0,101,20))
                # axes1.axhline(0.05,color='g',linewidth=0.4)
                # axes1.axhline(0.95,color='g',linewidth=0.4)
                #axes1.set_title(station_name + ' ' + station_channel,fontdict={'fontsize': 8,'color':'blue'})
               
                
        
        
        return fig,axes,#fig1,axes1
    
    ### Plotting spectra per each subset ###
    
    fig,axes= intensity_subplot(files_set,t0,stations_data)                                   
    fig.suptitle('ID Evento: igepn2023fkei Tiempo Origen(UTC): {}'.format(t0)+'\n Aceleraciones Componente Vertical')
    # fig.supylabel(r'Intensity')
    # fig.supxlabel(r'Time')
    fig.tight_layout(pad=1.75)
    fig.savefig(file_dir,dpi=700)
    # fig1.suptitle('Event: igepn2023fkei Time: {} \n Acceleration from event origin'.format(t0))
    # fig1.supylabel(r'Acceleration')
    # fig1.supxlabel(r'Time')
    # fig1.tight_layout(pad=1.25)
    # fig1.savefig(file_dir_acc,dpi=600) 
        
    

    
# class Earthquake_analysis():
    
#     def __init__(self,data_folder_path,wave_picking_file,stations_file,t_origin,magnitude,depth):
#         self.t0  = t_origin
#         self.wave_picking_csv =wave_picking_file
#         self.magnitude = magnitude
#         self.depth = depth 
#         self.files = os.listdir(data_folder_path)
        
    
        
#     def wavePick(self):
#         self.
        
        
    
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
    
    
    
    
 
    