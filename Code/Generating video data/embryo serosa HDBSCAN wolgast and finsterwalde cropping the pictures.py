# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 12:38:42 2021

@author: ryans
"""

import matplotlib.pyplot as plt
from skimage import io
import tribolium_clustering as tc

output_folder = tc.get_folder_path('location')






from skimage import io

for i in range(32):
    crop = io.imread(output_folder+'HDBSCAN embryo serosa rotation0 t{}.tif'.format(i))[22:884,461:929]
    io.imsave(output_folder+       'cropped HDBSCAN embryo serosa rotation0 t{}.tif'.format(i), crop)
    
    crop = io.imread(output_folder+'HDBSCAN embryo serosa rotation0 t{}.tif'.format(i))[22:884,461:929]
    io.imsave(output_folder+       'cropped HDBSCAN embryo rotation0 t{}.tif'.format(i), crop)
    
    crop = io.imread(output_folder+'HDBSCAN embryo serosa rotation1 t{}.tif'.format(i))[22:884,480:910]
    io.imsave(output_folder+       'cropped HDBSCAN embryo rotation1 t{}.tif'.format(i), crop)
    
    crop = io.imread(output_folder+'HDBSCAN embryo serosa rotation1 t{}.tif'.format(i))[22:884,480:910]
    io.imsave(output_folder+       'cropped HDBSCAN embryo rotation1 t{}.tif'.format(i), crop)

