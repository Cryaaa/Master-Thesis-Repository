# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 12:03:30 2021

@author: ryans
"""

import pyclesperanto_prototype as cle
import beetlesafari as bs 
import numpy as np

import tribolium_clustering as tc
from skimage import io
import pandas as pd


def regionprops_scikit_image(labelimage,originalimage):
        from skimage.measure import regionprops_table
        import pandas as pd
        import numpy as np
        # get lowest label index to adjust sizes of measurement arrays
        min_label = int(np.min(labelimage[np.nonzero(labelimage)]))
   
        #  defining function for getting standarddev as extra property
        # arguments must be in the specified order, matching regionprops
        def image_stdev(region, intensities):
            # note the ddof arg to get the sample var if you so desire!
            return np.std(intensities[region], ddof=1)
    
        # get region properties from labels
        regionprops = regionprops_table(labelimage.astype(dtype = 'uint16'), intensity_image= originalimage, 
                                        properties= ('area','bbox_area','convex_area','equivalent_diameter',
                                                     'euler_number','extent','feret_diameter_max','filled_area',
                                                     'major_axis_length','minor_axis_length', 'max_intensity',
                                                     'mean_intensity', 'min_intensity','solidity','centroid',
                                                     'weighted_centroid'),extra_properties=[image_stdev])
        
    
        print('Regionprops Completed')
    
    
    
        return pd.DataFrame(regionprops)

def topology_regionprops(gpu_output, nearest_neigh_list = [4,5,6], local = False):
    
    # Initialisation
    import pyclesperanto_prototype as cle
    import pandas as pd
    
    topology_props = {}
    
    avg_dmap_values = []
    stdev_dmap_values = []
    
    if local:
        loc_avg_dmap_values = []
        loc_stdev_dmap_values = []
    
    for n_count in nearest_neigh_list:
        # calculating the average distance of n nearest neighbours
        temp_avg_distance_map = cle.average_distance_of_n_closest_neighbors_map(gpu_output, n = n_count)
        temp_stdev_distance_map = cle.standard_deviation_of_touching_neighbors_map(temp_avg_distance_map, gpu_output, radius=2)
        
        # getting the values
        temp_avg_dmap_val = cle.read_intensities_from_map(gpu_output, temp_avg_distance_map)
        temp_stdev_dmap_val = cle.read_intensities_from_map(gpu_output, temp_stdev_distance_map)
        
        
        temp_stdev_distance_map = None
        
        #saving in lists for later processing
        avg_dmap_values.append(cle.pull(temp_avg_dmap_val)[0])
        stdev_dmap_values.append(cle.pull(temp_stdev_dmap_val)[0])
        temp_avg_dmap_val = None
        temp_stdev_dmap_val = None
        if local:
            temp_loc_avg_dmap = cle.mean_of_touching_neighbors_map(temp_avg_distance_map, gpu_output)
            
            temp_loc_stdev_dmap = cle.standard_deviation_of_touching_neighbors_map(temp_loc_avg_dmap, gpu_output, radius=2)
            
            temp_loc_avg_dmap_val = cle.read_intensities_from_map(gpu_output, temp_loc_avg_dmap)
            temp_loc_stdev_dmap_val = cle.read_intensities_from_map(gpu_output, temp_loc_stdev_dmap)
            
            temp_loc_avg_dmap = None
            temp_loc_stdev_dmap = None
            
            loc_avg_dmap_values.append(cle.pull(temp_loc_avg_dmap_val)[0])
            loc_stdev_dmap_values.append(cle.pull(temp_loc_stdev_dmap_val)[0])
            
            temp_loc_avg_dmap_val = None
            temp_loc_stdev_dmap_val = None
            
        temp_avg_distance_map = None
    if local:
        for avg_values, stdev_values, i in zip(loc_avg_dmap_values,loc_stdev_dmap_values,nearest_neigh_list):
            topology_props['local avg distance of {} closest points'.format(i)] = avg_values[1:]
            topology_props['local stddev distance of {} closest points'.format(i)] = stdev_values[1:]
            
    else:
        for avg_values, stdev_values, i in zip(avg_dmap_values,stdev_dmap_values,nearest_neigh_list):
            topology_props['avg distance of {} closest points'.format(i)] = avg_values[1:]
            topology_props['stddev distance of {} closest points'.format(i)] = stdev_values[1:]
            
    touch_matrix = cle.generate_touch_matrix(gpu_output)

    # ignore touching the background
    cle.set_column(touch_matrix,0,0)
    cle.set_row(touch_matrix,0,0)
    
    
    # detect touching neighbor count   
    touching_neighbor_count = cle.count_touching_neighbors(touch_matrix)
    touching_n_count = cle.pull(touching_neighbor_count)[0][1:]
    
    touching_neighbor_count = None
    touch_matrix = None
    gpu_output = None
    topology_props['touching neighbor count']= touching_n_count
    
    return pd.DataFrame(topology_props)

cle.select_device('GTX')

cc_dataset = bs.ClearControlDataset('F://2019-11-13-12-26-11-88-Wolgast_Tribolium_nGFP_TMR')
voxel_size = cc_dataset.get_voxel_size_zyx()

timeslice_duration = bs.hours_to_seconds(3)/32
start_time = bs.hours_to_seconds(11)
gastrulation_times = [start_time + (i+1) * timeslice_duration for i in range(32)]
print('beginning: {} hours'.format(bs.seconds_to_hours(gastrulation_times[0])))
print('ending   : {} hours'.format(bs.seconds_to_hours(gastrulation_times[-1])))

savedata_location = 'D://Uni//MSTER TUD//Master Thesis//output data//Wolgast Gastrulation Labels//'

# quick test to see if everything works
'''
workflow = io.imread(savedata_location + 'workflow t = {}s.tif'.format(gastrulation_times[0]))
tribolium = cc_dataset.get_image(time_in_seconds=gastrulation_times[0])
rescaled = tc.resample_isotropic_nodownsample(tribolium, voxelsize_yzx = voxel_size)

regprops = regionprops_scikit_image(workflow, rescaled)
topoprops = topology_regionprops(cle.push(workflow))

propscombined = pd.concat([regprops,topoprops], axis = 1, ignore_index=True)
'''

for i in gastrulation_times:
    workflow = io.imread(savedata_location + 'workflow t = {}s.tif'.format(i))
    tribolium = cc_dataset.get_image(time_in_seconds=i)
    rescaled = tc.resample_isotropic_nodownsample(tribolium, voxelsize_yzx = voxel_size)
    
    regprops = regionprops_scikit_image(workflow, rescaled)
    topoprops = topology_regionprops(cle.push(workflow))
    propscombined = pd.concat([regprops,topoprops], axis = 1, ignore_index=True)
    
    propscombined.to_csv(savedata_location + 'complete regionprops of t = {}s.csv'.format(i))
  