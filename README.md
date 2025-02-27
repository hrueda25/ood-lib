# Out of Domain (OOD) Detection Library

This repository provides a collection of methods for Out-of-Domain (OOD) detection implemented in Python. The library includes implementations of various algorithms for identifying data points that differ significantly from the training distribution.

## Features

- Multiple OOD detection methods:
  - One-Class SVM
  - P-Sphere Hull
- Standardized pipeline for comparing methods
- Batch processing of multiple datasets
- Comprehensive metrics reporting

## Installation

```bash
# Clone the repository
git clone https://github.com/hrueda25/ood-lib.git
cd ood-lib

# Install dependencies
pip install -r requirements.txt
```

## Usage Examples

Below are various ways to run the OOD detection pipeline with different configurations.

### Running a Single Method on a Single Dataset

```bash
# Run One-Class SVM on the abalone dataset
python src/ood_pipeline.py --method ocsvm --dataset abalone

# Run P-Sphere Hull on the iris dataset
python src/ood_pipeline.py --method psphere --dataset iris
```

### Running All Methods on a Single Dataset

```bash
# Run all implemented methods on the abalone dataset
python src/ood_pipeline.py --method all --dataset abalone
```

### Running a Method on All Datasets

```bash
# Run One-Class SVM on all datasets in the data folder
python src/ood_pipeline.py --method ocsvm --dataset all
```

### Customizing Method Parameters

```bash
# Customize One-Class SVM parameters
python src/ood_pipeline.py --method ocsvm --dataset abalone --nu 0.05 --kernel sigmoid

# Customize P-Sphere Hull parameters
python src/ood_pipeline.py --method psphere --dataset iris --n_clusters 50 --ps_vratio_filt
```

### Adjusting Train-Test Split

```bash
# Change the test set size to 30%
python src/ood_pipeline.py --method ocsvm --dataset abalone --test_size 0.3
```

### Specifying Output Location

```bash
# Save results to a specific file
python src/ood_pipeline.py --method all --dataset abalone --output results/my_experiment.csv
```

### Running a Comprehensive Experiment

```bash
# Run all methods on all datasets with custom parameters
python src/ood_pipeline.py --method all --dataset all --nu 0.05 --kernel rbf --n_clusters 50
```

## Data Format

The pipeline expects datasets in ARFF format, which can be placed in the `data/raw` folder. For datasets with missing values, the pipeline will handle them by replacing NaN values with mean values.

## Outputs

Results are saved as CSV files in the `results` folder. The filenames include the method and dataset information, along with key parameters used in the experiment.

Example output filename:
```
ood_results_dataset-iris_method-ocsvm_nu-0.1_kernel-rbf_test-0.2.csv
```

The standardized data is also saved in HDF5 format in the `data/input` folder for future use.

## Adding New Methods

To add a new OOD detection method:

1. Create a new file in the `src/methods` directory
2. Implement your method with the standard API similar to existing methods
3. Update the main pipeline `src/ood_pipeline.py` to include your new method

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.

## Citation

If you use this library in your research, please cite:

```
@software{ood_library,
  author = {Héctor Rueda and Ben Mathiensen},
  title = {Out of Domain Detection Library},
  year = {2025},
  url = {https://github.com/hrueda25/ood-lib}
}
```