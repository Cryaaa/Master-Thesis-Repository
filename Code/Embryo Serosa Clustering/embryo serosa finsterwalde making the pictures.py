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

            viewer.screenshot(save_data_location + name +' t{}'.format(timepoint) + ' rotation{} .tif'.format(i))

            time_in_msec = 1000
            QTimer().singleShot(time_in_msec, app.quit)
            viewer.close()

def embryo_serosa_preprocessing(regprops):
    from sklearn.preprocessing import StandardScaler
    import umap
    
    subselection = ['area', 'bbox_area', 'extent', 'feret_diameter_max', 'max_intensity',
                    'mean_intensity', 'min_intensity', 'solidity', 'centroid-0',
                    'centroid-1', 'centroid-2', 'image_stdev',
                    'avg distance of 6 closest points',
                    'stddev distance of 6 closest points', 'touching neighbor count',
                    'aspect_ratio']
    subselected = regprops[subselection]
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subselected)
    
    umap_object = umap.UMAP(
                         n_neighbors=400,
                         min_dist=0.0,
                         )

    processed_embedding = umap_object.fit_transform(scaled)
    
    return processed_embedding


finsterwalde = tc.processed_dataset(tc.get_folder_path('finsterwalde'))

finsterwalde_embedding = np.load('finsterwalde embryo serosa embedding.npy')
finsterwalde_prediction = np.load('finsterwalde embryo serosa presdiction.npy')




image_output_folder = 'C:/Users/ryans/OneDrive/Documents/Master Thesis/Documents/Figures/Embryo Serosa Clustering Finsterwalde//'
rotations_finsterwalde = [(0,130,0),(0,180,0),(0,230,0)]
cmap_napari = tc.napari_label_cmap()

            
for time in [0]:
    nice_screenshots_of_1_timepoint(finsterwalde,finsterwalde_prediction,
                                    time,cmap_napari,image_output_folder, 
                                    'embryo serosa clustering UMAP(nn 400, mindist 0), hdbscan(mincl size 2000)'
                                    ,rotations_finsterwalde)  
