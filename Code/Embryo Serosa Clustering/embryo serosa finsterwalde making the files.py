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


finsterwlade = tc.processed_dataset(tc.get_folder_path('finsterwlade'))
complete_regionprops_finsterwlade = finsterwlade.get_combined_regionprops()
complete_regionprops_finsterwlade = tc.min_maj_ax_to_aspectr(complete_regionprops_finsterwlade, del_min_maj=False)

finsterwlade_embedding = embryo_serosa_preprocessing(complete_regionprops_finsterwlade)

clusterer_all = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, leaf_size=40,
                                metric='euclidean', min_cluster_size=2000, 
                                min_samples=None, p=None)

finsterwlade_prediction = clusterer_all.fit_predict(finsterwlade_embedding)
np.save('finsterwalde embryo serosa embedding.npy',finsterwlade_embedding)
np.save('finsterwalde embryo serosa presdiction.npy',finsterwlade_prediction)


timepoint = 1
label_image = finsterwlade.get_labels(timepoint)
intensity_image = finsterwlade.get_intensity_image(timepoint)
cum_indices = finsterwlade.cumulative_label_lengths()

prediction = finsterwlade_prediction[cum_indices[timepoint]:cum_indices[timepoint+1]]
cluster_image = tc.generate_parametric_cluster_image(label_image,cle.push(label_image),prediction)

rot =(0,180,0)

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(intensity_image, rotate=rot)
viewer.add_labels(cluster_image, rotate=rot)

'''
image_output_folder = 'C:/Users/ryans/OneDrive/Documents/Master Thesis/Documents/Figures/Embryo Serosa Clustering Lund//'
rotations_wolgast = [(0,130,0),(0,180,0),(0,230,0)]
cmap_napari = tc.napari_label_cmap()

            
for time in range(26,32):
    nice_screenshots_of_1_timepoint(wolgast,wolgast_prediction,
                                    time,cmap_napari,image_output_folder, 
                                    'embryo serosa clustering UMAP(nn 400, mindist 0), hdbscan(mincl size 2000)'
                                    ,rotations_wolgast)  
'''