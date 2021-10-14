# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 15:24:40 2021

@author: ryans
"""
import tribolium_clustering as tc
import pyclesperanto_prototype as cle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import umap
import hdbscan
import napari


def nice_screenshots_of_1_timepoint(dataset, prediction_list, timepoint, 
                                    cmap, save_data_location, name, rotations):
    import napari
    import pyclesperanto_prototype as cle
    from qtpy.QtCore import QTimer
    
    label_image = dataset.get_labels(timepoint)
    intensity_image = dataset.get_intensity_image(timepoint)
    cum_indices = dataset.cumulative_label_lengths()
    
    prediction = prediction_list[cum_indices[timepoint]:cum_indices[timepoint+1]]
    prop = dataset.get_regionprops_timepoint(timepoint)

    regprop_with_predict = pd.concat([prop,pd.DataFrame(prediction, columns = ['prediction'],
                             index = prop.index)], axis = 1)
    regprop_with_predict.to_csv(save_data_location + 'regprops with ' + name +' t{}.csv'.format(timepoint))

    cluster_image = tc.generate_parametric_cluster_image(label_image,cle.push(label_image),prediction)
    
    for i,rot in enumerate(rotations):
        with napari.gui_qt() as app:
            viewer = napari.Viewer(ndisplay=3)
            viewer.add_image(intensity_image, rotate= rot)
            viewer.add_labels(cluster_image, rotate= rot, color = cmap)

            viewer.screenshot(save_data_location + name +' rotation{}'.format(i) + ' t{}.tif'.format(timepoint))

            time_in_msec = 1000
            QTimer().singleShot(time_in_msec, app.quit)
            viewer.close()



folder = 'D:/Uni/MSTER TUD/Master Thesis/output data/Wolgast Gastrulation Labels//'
wolgast = tc.processed_dataset(folder)

pred_location = 'C:/Users/ryans/OneDrive/Documents/Master Thesis/Documents/Figures/embryo serosa video files//'
wolgast_prediction_scaled = np.load(pred_location + 'wolgast_scaled_prediction.npy')
wolgast_prediction_unscaled = np.load(pred_location + 'wolgast_scaled_prediction.npy')



image_output_folder = pred_location + 'wolgast/'
rotations_wolgast = [(0,170,0),(0,0,0)]
cmap_napari = tc.napari_label_cmap()

#9, 18, 27
            
for time in range(18,27):
    nice_screenshots_of_1_timepoint(wolgast,wolgast_prediction_unscaled,
                                    time,cmap_napari,image_output_folder, 
                                    'embryo serosa Kmeans unscaled'
                                    ,rotations_wolgast)  
