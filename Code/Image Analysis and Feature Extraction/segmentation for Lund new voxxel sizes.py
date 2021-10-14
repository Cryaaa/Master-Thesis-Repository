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
        cle.erode_sphere(flop, flap)
        cle.erode_box(flap, flop)
        
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


cle.select_device('GTX')

cc_dataset = bs.ClearControlDataset(tc.get_folder_path('Lund CC Dataset Location'))
savedata_location = tc.get_folder_path('Save location')


voxel_size = cc_dataset.get_voxel_size_zyx()
alltrib = io.imread('D:/Uni/MSTER TUD/Master Thesis/First Coding Tries/tribolium_label_classification/Lund_18.0_22.0_Hours.tif')


for i, tribolium in enumerate(alltrib):
    rescaled = resample_isotropic_nodownsample(tribolium, voxel_size)
    output, gpu_output = workflow(rescaled)
    io.imsave(savedata_location + 'workflow t = {}s.tif'.format(i), output.astype('uint16'))
    io.imsave(savedata_location + 'rescaled t = {}s.tif'.format(i), cle.pull(rescaled))
 