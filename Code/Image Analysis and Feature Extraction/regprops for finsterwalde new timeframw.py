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

def testing_scikit_props(labelimage, originalimage):
    from skimage.measure import regionprops
    props = ('area','bbox_area','convex_area','equivalent_diameter',
             'euler_number','extent','feret_diameter_max','filled_area',
             'major_axis_length','minor_axis_length', 'max_intensity',
             'mean_intensity', 'min_intensity','solidity','centroid',
             'weighted_centroid')
    
    def image_stdev(region, intensities):
            # note the ddof arg to get the sample var if you so desire!
            return np.std(intensities[region], ddof=1)
    
    # get region properties from labels
    regionprops = regionprops(labelimage.astype(dtype = 'uint16'), 
                              intensity_image= originalimage, properties= props ,
                              extra_properties=[image_stdev])
    regpropdict ={}
    for i,region in enumerate(regionprops):
        for prop in props:
            try:
                regpropdict[prop] = region[prop]
            except:
                print('region number {} caused an error'.format(i) + 'with property {}'.format(prop))

def resample_isotropic_nodownsample(image_3d,voxelsize_yzx):
    import numpy as np
    import pyclesperanto_prototype as cle
    # we need to select a powerful GPU for this
    cle.select_device("GTX")
    if voxelsize_yzx[2] == voxelsize_yzx[1]:
        voxelsize_yzx = np.array(voxelsize_yzx)
        norm_voxelsize = voxelsize_yzx/voxelsize_yzx[2]
        input_image = cle.push_zyx(image_3d)
        resampled_image = cle.resample(input_image, factor_x=norm_voxelsize[2], 
                                       factor_y=norm_voxelsize[1], factor_z=norm_voxelsize[0])
    
        return resampled_image

def workflow(image, number_of_dilations = 15, number_of_erosions = 9):
    import numpy as np
    import pyclesperanto_prototype as cle    

    gpu_input = image

    # Spot detection
    # After some noise removal/smoothing, we perform a local maximum detection

    # gaussian blur -> needs adjusting, maybe even other filters for preprocessing
    gpu_tophat = cle.top_hat_sphere(gpu_input,radius_x=7, radius_y=7, radius_z=7)
    gpu_blurred = cle.gaussian_blur(gpu_tophat, sigma_x=1, sigma_y=1, sigma_z=2) 
    gpu_input = None
    # detect maxima: instead of a pointslist we get and image with white pixels at the maxima locations
    gpu_detected_maxima = cle.detect_maxima_box(gpu_blurred)
    gpu_tophat = None
    # Spot curation
    # Now, we remove spots with values below a certain intensity and label the remaining spots

    # threshold
    gpu_thresholded = cle.threshold_otsu(gpu_blurred)
    gpu_blurred = None

    # mask
    gpu_masked_spots = cle.mask(gpu_detected_maxima, gpu_thresholded)
    gpu_detected_maxima = None
    gpu_thresholded = None
    # label spots
    gpu_labelled_spots = cle.connected_components_labeling_box(gpu_masked_spots)
    gpu_masked_spots = None
    
    number_of_spots = cle.maximum_of_all_pixels(gpu_labelled_spots)
    print("Number of detected spots: " + str(number_of_spots))
    # retrieve the image to take a look at the maxima in napari
    # label map closing

    flip = cle.create_like(gpu_labelled_spots)
    flop = cle.create_like(gpu_labelled_spots)
    flag = cle.create([1,1,1])
    cle.copy(gpu_labelled_spots, flip)

    for i in range (0, number_of_dilations) :
        cle.onlyzero_overwrite_maximum_box(flip, flag, flop)
        cle.onlyzero_overwrite_maximum_diamond(flop, flag, flip)
    
    gpu_labelled_spots = None
    
    flap = cle.greater_constant(flip, constant= 1)

    for i in range(0, number_of_erosions):
        cle.erode_box(flap, flop)
        cle.erode_sphere(flop, flap)

    gpu_labels = cle.mask(flip, flap)
    flip = None
    flop = None
    flap = None
    flag = None
    
    alllabels = cle.close_index_gaps_in_label_map(gpu_labels)
    gpu_labels = None
    
    labels3d = only3dlabels(alllabels, image)
    alllabels = None
    
    output = cle.pull(labels3d)
    print('Label Numbering Starts at {val}'.format(val = np.min(output[np.nonzero(output)])))
    print('Workflow Completed')
    return output, labels3d

def only3dlabels(gpu_label_image,original_image):
    import pyclesperanto_prototype as cle
    import numpy as np
    
    cleregionprops = cle.statistics_of_background_and_labelled_pixels(original_image, gpu_label_image)

    bboxheight = cleregionprops['bbox_height']
    bboxwidth = cleregionprops['bbox_width']
    bboxdepth = cleregionprops['bbox_depth']
    bboxdiffx = cleregionprops['bbox_max_x'] - cleregionprops['bbox_min_x']
    bboxdiffy = cleregionprops['bbox_max_y'] - cleregionprops['bbox_min_y']
    bboxdiffz = cleregionprops['bbox_max_z'] - cleregionprops['bbox_min_z']
    
    flaglist =[]
    for i in range(int(len(bboxdepth))):
        if (bboxheight[i] <= 1 or bboxdepth[i] <= 1 or bboxwidth[i] <= 1 or 
            bboxdiffx[i] <=1 or bboxdiffy[i]<=1 or bboxdiffz[i] <= 1):
            flaglist.append(1)
        else:
            flaglist.append(0)
            
    flaglist_np = np.array(flaglist)
    
    deletedinstances = np.count_nonzero(flaglist_np)
    
    newflaglist = np.zeros(shape = flaglist_np.shape, dtype= 'uint16')
    count = 1
    for i in range(1, len(flaglist)):
        if (flaglist[i] == 0):
            newflaglist[i] = count
            count = count + 1
        else:
            newflaglist[i] = 0
    gpu_flaglist = cle.push(newflaglist)
    gpu_labels3d = cle.replace_intensities(gpu_label_image,gpu_flaglist)
    gpu_flaglist = None
    gpu_label_image = None
    print('{} deleted Objects'.format(deletedinstances))
    return gpu_labels3d

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
                                                     'major_axis_length','minor_axis_length', 'max_intensity', 'mean_intensity', 'min_intensity',
                                                     'solidity','centroid',
                                                     'weighted_centroid'),extra_properties=[image_stdev])
        #  extra_properties=[image_stdev]
    
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


import pandas as pd
cle.select_device('GTX')
savedata_location = tc.get_folder_path('Finsterwalde gastrulation labels')

timeslice_duration = bs.hours_to_seconds(3)/32
start_time = bs.hours_to_seconds(13.75)
gastrulation_times = [start_time + (i+1) * timeslice_duration for i in range(32)]
print('beginning: {} hours'.format(bs.seconds_to_hours(gastrulation_times[0])))
print('ending   : {} hours'.format(bs.seconds_to_hours(gastrulation_times[-1])))


for i in gastrulation_times:
    rescaled = io.imread(savedata_location + 'rescaled t = {}s.tif'.format(i))
    output = io.imread(savedata_location + 'workflow t = {}s.tif'.format(i))
    
    regprops = regionprops_scikit_image(output, rescaled)
    topoprops = topology_regionprops(cle.push(output))
    propscombined = pd.concat([regprops,topoprops], axis = 1, ignore_index=True)
    
    propscombined.to_csv(savedata_location + 'complete regionprops of t = {}s.csv'.format(i))
