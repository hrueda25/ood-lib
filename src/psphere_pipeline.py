
"""
psphere_pipeline.py
========
Contains functions related to the p-sphere hull method.
"""

import os
import sys
import numpy as np
from sklearn import cluster

# Add the path of the submodule where PSphereHull is located
module_path = os.path.abspath('../code/pspherehull/src')
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the necessary functions
from PSphereHull import refine_cluster_set as rcs
from PSphereHull import psphere_hull as psh

def hull_creation(scaled_features_train, n_clusters=100, ps_vratio_filt=True):
    """
    Creates the hull using the p-sphere hull method.
    
    Arguments
    ---------
    scaled_features_train : array-like
        Scaled feature data
    n_clusters : int, optional
        The number of clusters to form. Default is 100.
    ps_vratio_filt : bool, optional
        Wheter to filter based on the ps_vratio value. Default is True.

    Returns
    -------
    hull : PSphereHull object
    Calculated hull for the input data.

    Notes
    -----
    This function performs KMeans clustering on the input data, 
    describes the clusters' p-spheres, and filters based on 
    the ps_vratio value to create the final hull representation.
    """

    # Example using KMeans
    km = cluster.KMeans(n_clusters, random_state=42)
    km.fit(scaled_features_train)

    # Describe the p-spheres of the clusters
    spheres_df = rcs.describe_cluster_pspheres(scaled_features_train, km.labels_, km.cluster_centers_)

    # Filtering based on the ps_vration
    new_labels, new_centers = km.labels_, km.cluster_centers_

    psv = np.log10(spheres_df['ps_vratio'])

    if ps_vratio_filt:
        mean_psv = np.mean(psv)
        std_psv = np.std(psv)
        for cluster_index in psv.index:
            if psv[cluster_index] >= mean_psv + std_psv: 
                print(cluster_index, psv[cluster_index])
                new_labels, new_centers, success = rcs.split_cluster(scaled_features_train, new_labels, new_centers, cluster_index, min_size=4, verbose=True)
                new_labels = rcs.flag_outlier(scaled_features_train, new_labels, new_centers, cluster_index)
                

    # Redefine the p-spheres with new labels and centers
    spheres_df = rcs.describe_cluster_pspheres(scaled_features_train, new_labels, new_centers)

    # Create the hull
    hull = psh.PSphereHull(scaled_features_train, new_labels, new_centers, compute_all=False)
    hull.make_local_dimensions()
    hull.make_pcylinders()

    return hull