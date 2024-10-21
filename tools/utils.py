"""
utils.py
========
Contain utility functions for data processing
"""

from scipy.io import arff
import pandas as pd
import openml


def arff_to_df(arff_file, cat_to_num=True,):
    """
    Converts an ARFF file to a pandas DataFrame retrieving features and target.

    Arguments
    ---------
    arff_file : str
        Path to file in .arff format (OpenML data format)
    cat_to_num: bool, optional
        Wheter to convert categorical values to numerical (default is True)

    Returns
    -------
    features : pd.DataFrame
        DataFrame containing the feature columns
    target : pd.Series
        Series containing the target column
    """
    # Load the ARFF file
    data, meta = arff.loadarff(arff_file)

    # Convert the loaded ARFF data to a DataFrame
    df = pd.DataFrame(data)

    # The target is assumed to be the last column
    features = df.iloc[:, :-1]
    target = df.iloc[:, -1]

    # Optionally convert categorical columns to numerical codes
    if cat_to_num:
        cat_columns = features.select_dtypes(['object']).columns
        features[cat_columns] = features[cat_columns].apply(lambda col: col.str.decode('utf-8').astype('category').cat.codes)

    return features, target


def openml_to_df(dataset_name, cat_to_num=True,):
    """
    Converts an OpenML dataset to a pandas DataFrame retrieving features and target. 

    Arguments
    ---------
    dataset_name : str
        The name or ID of the OpenML dataset
    cat_to_num : bool, optional
        Whether to convert categorical values to numerical (default is True)

    Returns
    -------
    features : pd.DataFrame
        DataFrame containing the feature columns
    target : pd.Series
        Series containing the target column
    """
    # Get dataset by name or ID
    dataset = openml.datasets.get_dataset(dataset_name)

    # Get the data itself as a DataFrame
    features, target, _, _ = dataset.get_data(dataset_format="dataframe")

    # Optionally convert categorical columns to numerical codes
    if cat_to_num:
        cat_columns = features.select_dtypes(['object']).columns
        features[cat_columns] = features[cat_columns].apply(lambda col: col.str.decode('utf-8').astype('category').cat.codes)

    return features, target 

                  
