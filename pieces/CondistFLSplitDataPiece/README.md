# CondistFL Split Data Piece

Splits multi-organ segmentation datasets into k-fold cross-validation sets for federated learning.

## Description

Takes the four organ datasets from the DataLoader piece and creates fold-specific train/validation splits. Uses symlinks for NIfTI files to avoid data duplication. Each fold directory contains a `datalist.json` with proper `training` and `validation` arrays.

## Inputs

- **kidney_data_path**: Path to kidney dataset (from upstream DataLoader)
- **liver_data_path**: Path to liver dataset (from upstream DataLoader)
- **pancreas_data_path**: Path to pancreas dataset (from upstream DataLoader)
- **spleen_data_path**: Path to spleen dataset (from upstream DataLoader)
- **num_folds**: Number of CV folds (default: 3)
- **fold_index**: Which fold to use as validation, 0-based (default: 0)

## Outputs

- **kidney_data_root**: Fold-specific kidney dataset path with datalist.json
- **liver_data_root**: Fold-specific liver dataset path with datalist.json
- **pancreas_data_root**: Fold-specific pancreas dataset path with datalist.json
- **spleen_data_root**: Fold-specific spleen dataset path with datalist.json
- **fold_index**: The fold index used
- **num_folds**: Total number of folds
- **message**: Summary of the split

## Split Details

With 13-14 samples per dataset and 3 folds:
- Each fold: ~4-5 validation samples, ~8-9 training samples
- Fixed random seed (42) for reproducible shuffling
- Samples are deduplicated from the original datalist before splitting
