# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:09:05 2023

@author: fotogrametria
"""

import re


station = 'ISPG'
comp = 'N'


def url_finder(x,station,comp):
    pattern = fr'{station}+[^a-zA-Z0-9_]+(HN|BL|HH){comp}'
    re_pattern = re.compile(pattern)
    if  re.findall(re_pattern, x):
        print(x)
    
    
    
url_finder('D:\Proyectos\Earthquake_version\Earthquake_March18\data\EC.ISPG..HNE_20230318T171235.SAC',station,comp)