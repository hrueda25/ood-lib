"""
utils.py
========
Contains utility functions for data handling
"""

from scipy.io import arff
import pandas as pd
import openml


def arff_to_df(arff_file):
    """
    Converts an ARFF file to a pandas DataFrame retrieving features and target.

    Arguments
    ---------
    arff_file : str
        Path to file in .arff format (OpenML data format)

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

    return features, target


def openml_to_df(dataset_name):
    """
    Converts an OpenML dataset to a pandas DataFrame retrieving features and target. 

    Arguments
    ---------
    dataset_name : str
        The name or ID of the OpenML dataset

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
    features, target, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)

    return features, target 


def cat_to_num(df):
    """
    Converts categorical columns in a DataFrame to numerical values.

    Arguments
    ---------
    df : pd.DataFrame
        DataFrame containing categorical columns.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with categorical columns converted to numerical values.
    """      
    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda col: col.cat.codes)

    return df