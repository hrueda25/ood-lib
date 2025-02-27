"""
ood_pipeline.py
===============
Pipeline to run different Out-of-Domain (OOD) detection methods
on OpenML datasets
"""

import argparse
import openml
import numpy as np
import pandas as pd
import time
import os
import glob
import h5py
from datetime import datetime
from sklearn.model_selection import train_test_split

# Import preprocessing functions
from src.preproc.utils import arff_to_df, openml_to_df, cat_to_num
from src.preproc.basic import data_standardization  

# Import OOD methods
from src.methods.one_class_svm import run_one_class_svm
from src.methods.psphere_hull import hull_creation, hull_ood

def run_ocsvm(X_train, X_test, y_train=None, y_test=None, **kwargs):
    """Wrapper for One-Class SVM method"""
    print("[INFO] Running One-Class SVM...")
    
    # Extract parameters or use defaults
    nu = kwargs.get('nu', 0.1)
    kernel = kwargs.get('kernel', 'rbf')
    gamma = kwargs.get('gamma', 'scale')
    
    start_time = time.time()
    # Train on training data
    ood_labels = run_one_class_svm(X_train, X_test, nu=nu, kernel=kernel, gamma=gamma)
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics for test data
    test_ood = np.sum(ood_labels == -1)
    test_inliers = np.sum(ood_labels == 1)
    test_ood_ratio = test_ood / len(ood_labels) if len(ood_labels) > 0 else 0
    
    print(f"[INFO] Testing: OOD samples detected: {test_ood} ({test_ood_ratio:.2%})")
    
    result = {
        'method': 'ocsvm',
        'params': f"nu={nu},kernel={kernel},gamma={gamma}",
        'test_samples': len(ood_labels),
        'test_ood': test_ood,
        'test_inliers': test_inliers,
        'test_ood_ratio': test_ood_ratio,
        'runtime_seconds': elapsed_time
    }
    
    return result

def run_psphere(X_train, X_test, y_train=None, y_test=None, **kwargs):
    """Wrapper for p-sphere hull method"""
    print("[INFO] Running p-sphere hull...")
    
    # Extract parameters or use defaults
    n_clusters = kwargs.get('n_clusters', 100)
    ps_vratio_filt = kwargs.get('ps_vratio_filt', True)
    
    start_time = time.time()
    # Create hull based on training data
    hull = hull_creation(X_train, n_clusters=n_clusters, ps_vratio_filt=ps_vratio_filt)
    
    print("[INFO] Hull created.")
    
    # Determine which points are inside/outside the hull for test data
    test_points_inside, test_points_outside = hull_ood(hull, X_test)
    elapsed_time = time.time() - start_time
    
    # Calculate metrics for test data
    test_ood = len(test_points_outside)
    test_inliers = len(test_points_inside)
    test_ood_ratio = test_ood / len(X_test) if len(X_test) > 0 else 0
    
    print(f"[INFO] Testing: OOD samples detected: {test_ood} ({test_ood_ratio:.2%})")
    
    result = {
        'method': 'psphere',
        'params': f"n_clusters={n_clusters},ps_vratio_filt={ps_vratio_filt}",
        'test_samples': len(X_test),
        'test_ood': test_ood,
        'test_inliers': test_inliers,
        'test_ood_ratio': test_ood_ratio,
        'runtime_seconds': elapsed_time
        }
    
    return result


def get_local_datasets(data_folder='data/raw/openML/'):
    """Get a list of all ARFF datasets in the data folder"""
    
    # Get all .arff files in the data folder
    arff_files = glob.glob(os.path.join(data_folder, '*.arff'))
    
    # Extract just the filenames without extension
    dataset_names = [os.path.splitext(os.path.basename(file))[0] for file in arff_files]
    
    print(f"[INFO] Found {len(dataset_names)} ARFF datasets in {data_folder}: {', '.join(dataset_names)}")
    
    return arff_files


def main():

    # 1. Parse command-line arguments
    parser = argparse.ArgumentParser(description="OOD Methods Pipeline on OpenML data.")
    
    parser.add_argument(
        "--method",
        type=str,
        default="psphere",
        choices=["ocsvm", "psphere", "all"],
        help="Which OOD method to use (ocsvm, psphere, all)."
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="abalone",
        help="Name or ID of the dataset on OpenML, or 'all' for multiple datasets."
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Proportion of the dataset to include in the test split (default: 0.2)."
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="results/results.csv",
        help="Path to the output CSV file for results (default: results/results.csv)."
    )
    
    parser.add_argument(
        "--data_folder",
        type=str,
        default="data/raw/openML/",
        help="Path to folder containing ARFF datasets (default: data)."
    )
    
    # OCSVM specific parameters
    parser.add_argument("--nu", type=float, default=0.1, help="Parameter for OCSVM")
    parser.add_argument("--kernel", type=str, default="rbf", choices=["linear", "poly", "rbf", "sigmoid"], 
                        help="Kernel for OCSVM")
    parser.add_argument("--gamma", type=str, default="scale", help="Gamma parameter for OCSVM")
    
    # PSphere specific parameters
    parser.add_argument("--n_clusters", type=int, default=100, help="Number of clusters for p-sphere hull")
    parser.add_argument("--ps_vratio_filt", action="store_true", help="Filter based on ps_vratio")
    
    args = parser.parse_args()
    
    # Initialize results container
    results = []
    
    # 2. Determine which datasets to process
    datasets_to_process = []
    if args.dataset.lower() == "all":
        print(f"[INFO] Processing all datasets in {args.data_folder} folder...")
        datasets_to_process = get_local_datasets(args.data_folder)
    else:
        # Check if it's a full path or just a name
        if os.path.exists(args.dataset):
            datasets_to_process = [args.dataset]
        else:
            # Try to find it in the data folder
            potential_path = os.path.join(args.data_folder, f"{args.dataset}.arff")
            if os.path.exists(potential_path):
                datasets_to_process = [potential_path]
            else:
                print(f"[ERROR] Dataset {args.dataset} not found in {args.data_folder}")
                return
    
    # 3. Determine which methods to run
    methods_to_run = []
    if args.method.lower() == "all":
        methods_to_run = ["ocsvm", "psphere"]
    else:
        methods_to_run = [args.method]
    
    # 4. Process each dataset
    for dataset_name in datasets_to_process:
        try:
            print(f"\n[INFO] Processing dataset: {dataset_name}")
            
            # Load dataset from ARFF file
            dataset_path = dataset_name
            dataset_display_name = os.path.splitext(os.path.basename(dataset_path))[0]
            print(f"[INFO] Loading dataset from file: {dataset_path}")
            try:
                features, target = openml_to_df(dataset_display_name)
                features = cat_to_num(features)
                print(f"[INFO] Dataset dimensions: features={features.shape}, target={target.shape}")
            except Exception as e:
                print(f"[ERROR] Failed to load dataset {dataset_display_name}: {str(e)}")
                continue

            # Clean NaN values by replacing them with zeros or means
            features = features.fillna(features.mean())
            
            # Always standardize the data
            print("[INFO] Standardizing data...")
            try:
                scaled_features, scaled_targets = data_standardization(features, target)
                
                # Split data into training and testing sets
                print(f"[INFO] Splitting data with test_size={args.test_size}")
                X_train, X_test, y_train, y_test = train_test_split(
                    scaled_features, scaled_targets, 
                    test_size=args.test_size, 
                    random_state=42
                )
                print(f"[INFO] Training set: {X_train.shape}, Test set: {X_test.shape}")
                
            except Exception as e:
                print(f"[ERROR] Failed to preprocess data: {str(e)}")
                continue

            # Save all standardized data in one HDF5 file
            h5_filepath = f'data/input/{dataset_display_name}_standardized.h5'
            if not os.path.exists(h5_filepath):
                # Create input directory if it doesn't exist
                os.makedirs('data/input', exist_ok=True)
                
                # Save to HDF5 file
                with h5py.File(h5_filepath, 'w') as f:
                    f.create_dataset('X_train', data=X_train)
                    f.create_dataset('X_test', data=X_test)
                    f.create_dataset('y_train', data=y_train)
                    f.create_dataset('y_test', data=y_test)
                print(f"[INFO] Standardized data saved to {h5_filepath}")
            else:
                print(f"[INFO] Standardized data file {h5_filepath} already exists, skipping save")
                        
            # Run each method
            for method in methods_to_run:
                try:
                    # Create method parameters dictionary
                    method_params = {}
                    
                    if method == "ocsvm":
                        method_params = {
                            'nu': args.nu,
                            'kernel': args.kernel,
                            'gamma': args.gamma
                        }
                        method_result = run_ocsvm(X_train, X_test, y_train, y_test, **method_params)
                    
                    elif method == "psphere":
                        method_params = {
                            'n_clusters': args.n_clusters,
                            'ps_vratio_filt': args.ps_vratio_filt
                        }
                        method_result = run_psphere(X_train, X_test, y_train, y_test, **method_params)
                    
                    # Add dataset info to results
                    method_result['dataset'] = dataset_display_name
                    method_result['dataset_path'] = dataset_path
                    method_result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Add to results list
                    results.append(method_result)
                    
                except Exception as e:
                    print(f"[ERROR] Failed to run method {method} on dataset {dataset_display_name}: {str(e)}")
    
        except Exception as e:
            print(f"[ERROR] An error occurred while processing dataset {dataset_name}: {str(e)}")
    
    
# 5. Save results to CSV in the results folder (assuming it exists)
    if results:
        try:
            results_df = pd.DataFrame(results)
            
            # Generate a filename with parser parameters instead of timestamp
            if os.path.basename(args.output) == "results.csv":
                # Include dataset info in the filename
                if args.dataset.lower() == "all":
                    dataset_str = "allDatasets"
                else:
                    # Extract just the dataset name without path or extension
                    dataset_str = os.path.splitext(os.path.basename(args.dataset))[0]
                
                # Create a string with the key parameters
                param_string = f"dataset-{dataset_str}_method-{args.method}"
                
                if args.method == "ocsvm" or args.method == "all":
                    param_string += f"_nu-{args.nu}_kernel-{args.kernel}"
                
                if args.method == "psphere" or args.method == "all":
                    param_string += f"_clusters-{args.n_clusters}"
                
                param_string += f"_test-{args.test_size}"
                
                # Replace special characters that would be invalid in filenames
                param_string = param_string.replace(" ", "_").replace("/", "-")
                
                # Create the full filename
                filename = f"ood_results_{param_string}.csv"
                args.output = os.path.join(os.path.dirname(args.output), filename)
            
            # Save to CSV
            results_df.to_csv(args.output, index=False)
            print(f"\n[INFO] Results saved to {args.output}")
        
        except Exception as e:
            print(f"[ERROR] Failed to save results: {str(e)}")
    else:
        print("[WARNING] No results to save.")
    
    print("\n[INFO] Pipeline finished.")

if __name__ == "__main__":
    main()