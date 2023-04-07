# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 19:18:38 2023

@author: joanv
"""

from obspy.signal.util import smooth
import matplotlib.pylab as plt
import obspy as obs
data = "./data/EC.AC07..HNZ_20230318T171237.SAC"
trace = obs.read(data)[0]
smoothed = smooth(trace,15)
plt.plot(trace[10000:15000],linewidth=0.4)
plt.plot(smoothed[10000:15000],'k')
plt.show()