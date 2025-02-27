"""
one_class_svm.py
================
Implements the One-Class SVM method for OOD detection.
"""

import numpy as np
from sklearn.svm import OneClassSVM

def run_one_class_svm(scaled_features_train, scaled_features_test, nu=0.1, kernel='rbf', gamma='scale'):

    """
    Trains and applies a One-Class SVM for OOD detection.

    Arguments
    ----------
    scaled_features_train : array-like
        Scaled feature data
    nu : float, default=0.1
        An upper bound on the fraction of training errors and a lower bound
        on the fraction of support vectors. Should be in the interval (0, 1].
    kernel : str, default='rbf'
        Specifies the kernel type to be used in the algorithm.
        It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'.
    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
        - 'scale' is 1 / (n_features \* X.var()) as value of gamma.
        - 'auto' uses 1 / n_features.

    Returns
    -------
    predictions : array, shape (n_samples,)
        The predicted labels for each sample:
        -1 indicates out-of-domain (anomaly).
         1 indicates in-domain (normal).
    
    Notes
    -----
    - You can adjust nu, kernel, and gamma for better results depending on your data.
    """

    # Create the One-Class SVM model

    oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)

    # Fit on the entire dataset (unsupervised OOD detection)

    oc_svm.fit(scaled_features_train)

    # Predict: -1 for outliers, 1 for inliers

    predictions = oc_svm.predict(scaled_features_test)

    return predictions