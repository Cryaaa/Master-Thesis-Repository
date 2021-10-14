# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 13:49:04 2021

@author: ryans
"""


def annotation_screenshots_of_1_timepoint(dataset, dataset_folder, timepoint,
                                          annotation_prefix,
                                          cmap, save_data_location, 
                                          rotation ,name):
    import tribolium_clustering as tc
    import napari
    from skimage import io
    import pyclesperanto_prototype as cle
    from qtpy.QtCore import QTimer
    
    label_image = dataset.get_labels(timepoint)
    intensity_image = dataset.get_intensity_image(timepoint)
    
    annotated_img = io.imread(dataset_folder + annotation_prefix + 't{}.tif'.format(timepoint))
    cle_regp = cle.statistics_of_background_and_labelled_pixels(annotated_img,
                                                                label_image)
    annotation_labels = cle_regp['max_intensity'][1:].astype('uint8')-1

    cluster_image = tc.generate_parametric_cluster_image(label_image,cle.push(label_image),annotation_labels)
    
    
    with napari.gui_qt() as app:
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(intensity_image, rotate= rotation)
        viewer.add_labels(cluster_image, rotate= rotation, color = cmap)

        viewer.screenshot(save_data_location + name +' t{}.tif'.format(timepoint))

        time_in_msec = 1000
        QTimer().singleShot(time_in_msec, app.quit)
        viewer.close()




import tribolium_clustering as tc
import numpy as np
import pyclesperanto_prototype as cle
from skimage import io

wolgast_folder = tc.get_folder_path('wolgast folder')
save_folder = tc.get_folder_path('save folder')
wolgast = tc.processed_dataset(wolgast_folder)

np_cmap = tc.napari_label_cmap()

wolgast_annot_tps = [0,3,6,9,12,15,20,31]

for i in wolgast_annot_tps:
    annotation_screenshots_of_1_timepoint(wolgast, wolgast_folder, i,
                                          'serosa embryo annotation rescaled ',
                                          np_cmap, save_folder, 
                                          rotation = (0,180,0) ,
                                          name = 'Embryo Serosa Annotation Image t{}.tif'.format(i))


