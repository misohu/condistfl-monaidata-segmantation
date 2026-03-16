# CondistFL Trainer Domino Piece

This Domino piece trains a federated learning model using the CondistFL framework with NVFlare.

## Description

Trains multi-organ and tumor segmentation models across multiple federated clients without sharing raw data. Uses conditional distillation to handle partially annotated datasets.

## Training Process

1. Configures data paths for each client
2. Launches NVFlare simulator
3. Trains for specified number of rounds
4. Saves best global model and client-specific models
5. Returns workspace with all training artifacts

## Inputs

- **num_rounds**: Number of FL rounds (default: 3)
- **steps_per_round**: Training steps per round (default: 1000)
- **clients**: Comma-separated client names (default: "liver,spleen,pancreas,kidney")
- **gpus**: GPU IDs to use (default: "0,1,2,3")
- **data_root_kidney**: Path to kidney dataset
- **data_root_liver**: Path to liver dataset
- **data_root_pancreas**: Path to pancreas dataset
- **data_root_spleen**: Path to spleen dataset
- **workspace_dir**: Output directory for training results

## Outputs

- **workspace_dir**: Directory with training logs, models, and metrics
- **best_global_model**: Path to best global model checkpoint
- **training_complete**: Success status
- **num_rounds_completed**: Rounds completed
- **message**: Status message

## Usage

1. Add piece to Domino workflow
2. Configure data paths and training parameters
3. Run workflow
4. Use output model with CondistFL Predictor piece

## Requirements

- 4 GPUs with 16GB+ VRAM each (or adjust GPU configuration)
- Training data for each organ type
- Adequate disk space for workspace (~50GB+)

## Output

The workspace contains:
- `app_server/` - Global server artifacts and best model
- `app_{client}/` - Client-specific models and logs
- TensorBoard logs for monitoring
