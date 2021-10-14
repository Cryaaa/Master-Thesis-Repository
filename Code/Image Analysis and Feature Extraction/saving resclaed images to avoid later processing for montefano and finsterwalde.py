# -*- coding: utf-8 -*-
"""
Created on Fri May 21 15:12:22 2021

@author: ryans
"""
# Initialize GPU
import pyclesperanto_prototype as cle
import beetlesafari as bs 
import numpy as np
from skimage.io import imread
import tribolium_clustering as tc
from skimage import io




'''
============================ Wolgast ========================================
'''

import pandas as pd
cle.select_device('GTX')


cc_dataset = bs.ClearControlDataset(tc.get_folder_path('Choose Wolgast Dataset Directory'))
savedata_location = tc.get_folder_path('Save Location')

print("Duration in seconds:", cc_dataset.get_duration_in_seconds())
print("Duration in hours:", bs.seconds_to_hours(cc_dataset.get_duration_in_seconds()))

timeslice_duration = bs.hours_to_seconds(3)/32
start_time = bs.hours_to_seconds(11)
gastrulation_times = [start_time + (i+1) * timeslice_duration for i in range(32)]
print('beginning: {} hours'.format(bs.seconds_to_hours(gastrulation_times[0])))
print('ending   : {} hours'.format(bs.seconds_to_hours(gastrulation_times[-1])))
voxel_size = cc_dataset.get_voxel_size_zyx()


test_time = [gastrulation_times[0]]
for i in gastrulation_times:
    rescaled = tc.resample_isotropic_nodownsample(cc_dataset.get_image(time_in_seconds=i)
                                                  , voxel_size)
    io.imsave(savedata_location + 'rescaled t = {}s.tif'.format(i), rescaled)
    print('dopamine')
   

rescaled = None
gpu_output = None

np.save(savedata_location + 'timepoints.npy', gastrulation_times)
#savedata_location = tc.get_folder_path('Save Location')
#np.save(savedata_location + 'rotation.npy', (-30,180,85))

'''
=============================== Finsterwalde =================================
'''
'''
import pandas as pd
cle.select_device('GTX')


cc_dataset = bs.ClearControlDataset('D://2019-10-28-17-22-59-23-Finsterwalde_Tribolium_nGFP')
savedata_location = 'C://Users//ryans//OneDrive//Documents//Master Thesis//output data//'


print("Duration in seconds:", cc_dataset.get_duration_in_seconds())
print("Duration in hours:", bs.seconds_to_hours(cc_dataset.get_duration_in_seconds()))

timeslice_duration = bs.hours_to_seconds(3)/32
start_time = bs.hours_to_seconds(13)
gastrulation_times = [start_time + (i+1) * timeslice_duration for i in range(32)]
print('beginning: {} hours'.format(bs.seconds_to_hours(gastrulation_times[0])))
print('ending   : {} hours'.format(bs.seconds_to_hours(gastrulation_times[-1])))
voxel_size = cc_dataset.get_voxel_size_zyx()


test_time = [gastrulation_times[0]]
for i in gastrulation_times:
    rescaled = cle.pull(cc_dataset.get_resampled_image(time_in_seconds=i))
    io.imsave(savedata_location + 'Finsterwalde Gastrulation Labels//rescaled t = {}s.tif'.format(i), rescaled)
    print('satisfaction')
'''    