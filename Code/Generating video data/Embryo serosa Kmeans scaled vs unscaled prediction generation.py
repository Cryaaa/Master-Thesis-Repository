# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 18:41:33 2021

@author: ryans
"""

import tribolium_clustering as tc
import pyclesperanto_prototype as cle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def embryo_serosa_preprocessing(regprops):
    from sklearn.preprocessing import StandardScaler
    aspectrat = tc.min_maj_ax_to_aspectr(regprops,del_min_maj=False)
    subselection = ['area', 'bbox_area', 'extent', 'feret_diameter_max', 'max_intensity',
                    'mean_intensity', 'min_intensity', 'solidity', 'centroid-0',
                    'centroid-1', 'centroid-2', 'image_stdev',
                    'avg distance of 6 closest points',
                    'stddev distance of 6 closest points', 'touching neighbor count',
                    'aspect_ratio']
    subselected = aspectrat[subselection]
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subselected)
    
    return scaled

def embryo_serosa_preprocessing_no_scaling(regprops):
    from sklearn.preprocessing import StandardScaler
    aspectrat = tc.min_maj_ax_to_aspectr(regprops,del_min_maj=False)
    subselection = ['area', 'bbox_area', 'extent', 'feret_diameter_max', 'max_intensity',
                    'mean_intensity', 'min_intensity', 'solidity', 'centroid-0',
                    'centroid-1', 'centroid-2', 'image_stdev',
                    'avg distance of 6 closest points',
                    'stddev distance of 6 closest points', 'touching neighbor count',
                    'aspect_ratio']
    subselected = aspectrat[subselection]
    
    return subselected

lund_folder = tc.get_folder_path('lund')
lund = tc.processed_dataset(lund_folder)
wolg_folder = tc.get_folder_path('wolgast')
wolg = tc.processed_dataset(wolg_folder)
fnst_folder = tc.get_folder_path('finsterwalde')
fnst = tc.processed_dataset(fnst_folder)

lund_cluster_props_scaled = embryo_serosa_preprocessing(lund.get_combined_regionprops())
lund_cluster_props = embryo_serosa_preprocessing_no_scaling(lund.get_combined_regionprops())
wolg_cluster_props_scaled = embryo_serosa_preprocessing(wolg.get_combined_regionprops())
wolg_cluster_props = embryo_serosa_preprocessing_no_scaling(wolg.get_combined_regionprops())
fnst_cluster_props_scaled = embryo_serosa_preprocessing(fnst.get_combined_regionprops())
fnst_cluster_props = embryo_serosa_preprocessing_no_scaling(fnst.get_combined_regionprops())

from sklearn.cluster import KMeans

kmeans_clust = KMeans(n_clusters=2,max_iter=10000, init= 'k-means++', random_state=42)

unscaled_pred_lund = kmeans_clust.fit_predict(lund_cluster_props)
scaled_pred_lund = kmeans_clust.fit_predict(lund_cluster_props_scaled)
unscaled_pred_wolg = kmeans_clust.fit_predict(wolg_cluster_props)
scaled_pred_wolg = kmeans_clust.fit_predict(wolg_cluster_props_scaled)
unscaled_pred_fnst = kmeans_clust.fit_predict(fnst_cluster_props)
scaled_pred_fnst = kmeans_clust.fit_predict(fnst_cluster_props_scaled)


unscaled_preds = (unscaled_pred_lund,unscaled_pred_wolg,unscaled_pred_fnst)
scaled_preds = (scaled_pred_lund,scaled_pred_wolg,scaled_pred_fnst)

output_folder = tc.get_folder_path('kmeans embryo serosa videos')
names =['lund','wolgast','finsterwalde']


for scaled,name in zip(scaled_preds,names):
    np.save(output_folder+'{}_scaled_prediction'.format(name),scaled)
    
    
for unscaled,name in zip(unscaled_preds,names):
    np.save(output_folder+'{}_unscaled_prediction'.format(name),unscaled)