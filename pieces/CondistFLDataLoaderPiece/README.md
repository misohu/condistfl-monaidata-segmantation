# CondistFL Data Loader Piece

Loads the baked-in multi-organ CT segmentation datasets for the CondistFL federated learning demo.

## Description

This piece ships with four organ-specific datasets (KiTS19/kidney, Liver, Pancreas, Spleen) baked into its Docker image. At runtime it copies them to shared storage so downstream pieces (SplitData, Trainer) can access the data.

## Inputs

- **data_dir**: Root directory containing dataset folders (default: `/data`)

## Outputs

- **kidney_data_path**: Path to the copied KiTS19 dataset
- **liver_data_path**: Path to the copied Liver dataset
- **pancreas_data_path**: Path to the copied Pancreas dataset
- **spleen_data_path**: Path to the copied Spleen dataset
- **message**: Status message

## Datasets

Each dataset folder contains:
- `datalist.json` — sample list with image/label pairs
- `*.nii.gz` — NIfTI image and label volumes (13-14 samples each)
