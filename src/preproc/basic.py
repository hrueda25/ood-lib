"""
basic.py
========
Contains basic functions for data preprocessing
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA

def data_standardization(features, target=None):
    """
    Standardizes the feature and target data using StandardScaler.

    Arguments
    ---------
    features : np.ndarray or pd.DataFrame
        Feature data to be standardized
    target : np.ndarray or pd.Series, optional
        Target data to be standardized. If None, only features will be scaled

    Returns
    -------
    scaled_features : np.ndarray
        Standardized feature data
    scaled_target : np.ndarray, optional
        Standardized target data, if target is provided.
    """
    ss = StandardScaler()

    # Standardize features
    scaled_features = ss.fit_transform(features)

    # Standardize target if provided
    if target is not None:
        scaled_target = ss.fit_transform(target.values.reshape(-1, 1)) # Reshape if target is a 1D array
        return scaled_features, scaled_target
    else:
        return scaled_features

 

def mask_0_features(features):
    """
    Removes features (columns) where all values are zero.

    Arguments
    ---------
    features : np.ndarray or pd.DataFrame
        Feature data.

    Returns
    -------
    filtered_features : np.ndarray
        Feature data with zero-value columns removed.
    """
    # Identify columns where all values are zero
    non_zero_columns = ~np.all(features == 0, axis=0)

    # Filter out columns with only zeros
    filtered_features = features[:, non_zero_columns]

    return filtered_features



def PCA_transform():
    """
    Arguments
    ---------

    Returns
    -------
    """
    transformer = SparsePCA(n_components=3, random_state=0)
    transformer.fit(scaled_features)

    return 