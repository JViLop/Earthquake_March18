# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:09:05 2023

@author: fotogrametria
"""

import re


station = 'ACH1'
comp = 'E'

@staticmethod
def url_finder(x,station,comp):
    pattern = fr'{station}+[^0-9]+{comp}'
    re_pattern = re.compile(pattern)
    if  re.findall(re_pattern, x):
        return 