"""

ood_pipeline.py

===============

Pipeline to run different Out-of-Domain (OOD) detection methods

on OpenML datasets.

"""

import argparse
import openml
import numpy as np

# Import preprocessing functions
from src.preproc.utils import openml_to_df, cat_to_num
from src.preproc.basic import data_standardization  

# Import OOD methods (adjust paths and names accordingly)
from src.methods.one_class_svm import run_one_class_svm
from src.methods.psphere_hull import hull_creation

# from src.methods.isolation_forest import run_isolation_forest  # Example for future methods

def main():

   # 1. Parse command-line arguments

   parser = argparse.ArgumentParser(description="OOD Methods Pipeline on OpenML data.")

   parser.add_argument(

       "--method",

       type=str,

       default="ocsvm",

       choices=["ocsvm", "psphere"],

       help="Which OOD method to use (ocsvm, psphere)."

   )

   parser.add_argument(

       "--dataset",

       type=str,

       default="abalone",

       help="Name or ID of the dataset on OpenML."

   )

   parser.add_argument(

       "--standardize",

       action="store_true",

       help="If set, data will be standardized before the OOD method."

   )

   args = parser.parse_args()

   # 2. Load the dataset from OpenML

   print(f"\n[INFO] Loading dataset '{args.dataset}' from OpenML...")

   features, target = openml_to_df(args.dataset)

   features = cat_to_num(features)

   print(f"[INFO] Dataset dimensions: features={features.shape}, target={target.shape}")

   # 3. Optional data preprocessing (e.g., standardization)

   if args.standardize:

       print("[INFO] Standardizing data...")

       print(features)
       scaled_features, scaled_targets = data_standardization(features, target)

   # 4. Choose which OOD method to run

   if args.method == "ocsvm":

       print("[INFO] Running One-Class SVM...")

       # Example usage of run_one_class_svm

       ood_labels = run_one_class_svm(scaled_features)

       # ood_labels should return an array of -1 for OOD and 1 for inliers

       num_ood = np.sum(ood_labels == -1)

       print(f"[INFO] OOD samples detected: {num_ood}")

   elif args.method == "psphere":

       print("[INFO] Running p-sphere hull...")

       # Example usage of hull_creation

       hull = hull_creation(scaled_features)

       print("[INFO] Hull created.")

       # implement OOD determination

   else:

       print("[ERROR] Unrecognized OOD method.")

   print("[INFO] Pipeline finished.")

if __name__ == "__main__":

   main()


