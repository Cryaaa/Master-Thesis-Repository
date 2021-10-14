# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 23:53:39 2021

@author: ryans
"""
import tribolium_clustering as tc
import pyclesperanto_prototype as cle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, v_measure_score, silhouette_score, calinski_harabasz_score














def nice_screenshots_of_1_timepoint(dataset, prediction_list, timepoint, 
                                    cmap, save_data_location, name, rotation):
    import napari
    import pyclesperanto_prototype as cle
    from qtpy.QtCore import QTimer
    import pandas as pd
    import tribolium_clustering as tc
    
    label_image = dataset.get_labels(timepoint)
    intensity_image = dataset.get_intensity_image(timepoint)
    cum_indices = dataset.cumulative_label_lengths()
    
    prediction = prediction_list[cum_indices[timepoint]:cum_indices[timepoint+1]]
    prop = dataset.get_regionprops_timepoint(timepoint)

    regprop_with_predict = pd.concat([prop,pd.DataFrame(prediction, columns = ['prediction'],
                             index = prop.index)], axis = 1)
    regprop_with_predict.to_csv(save_data_location + 'regprops with ' + name +' t{}.csv'.format(timepoint))

    cluster_image = tc.generate_parametric_cluster_image(label_image,cle.push(label_image),prediction)
    
    with napari.gui_qt() as app:
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(intensity_image, rotate= rotation)
        viewer.add_labels(cluster_image, rotate= rotation, color = cmap)

        viewer.screenshot(save_data_location + name +' t{}.tif'.format(timepoint))

        time_in_msec = 1000
        QTimer().singleShot(time_in_msec, app.quit)
        viewer.close()
        
lund_folder = tc.get_folder_path('lund')
lund = tc.processed_dataset(lund_folder)


pred_location = tc.get_folder_path('prediction location')
predictions = np.load(pred_location + 'predictions for cluster numbers 2 - 8.npy')[3]


output_folder = pred_location


np_cmap = tc.napari_label_cmap()

for i in range(31,32):
    nice_screenshots_of_1_timepoint(lund,predictions, i, 
                                    np_cmap, output_folder, 
                                    'K-means ++, 5 clusters, UMAP 7d, 400 neighbours , correlation filtered view from dorsal side', 
                                    rotation=(0,0,0))


from skimage import io
for i in range(26,32):
    crop = io.imread(output_folder+'K-means ++, 5 clusters, UMAP 7d, 400 neighbours , correlation filtered view from dorsal side t{}.tif'.format(i))[22:884,480:910]
    io.imsave(output_folder+'K-means ++, 5 clusters, UMAP 7d, 400 neighbours , correlation filtered view from dorsal side t{}.tif'.format(i), crop)




'''
from skimage import io
tp = 0
crop = io.imread(output_folder+'K-means ++, scaled, correlation filtered t{}.tif'.format(tp))[22:884,480:910]



for i in range(25):
    crop = io.imread(output_folder+'K-means ++, scaled, correlation filtered t{}.tif'.format(i))[22:883,445:945]
    io.imsave(output_folder+'K-means ++, scaled, correlation filtered t{}.tif'.format(i), crop)
    crop = io.imread(output_folder+'K-means ++, unscaled, correlation filtered t{}.tif'.format(i))[22:883,445:945]
    io.imsave(output_folder+'K-means ++, unscaled, correlation filtered t{}.tif'.format(i), crop)
'''