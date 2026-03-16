# CondistFL Trainer

Domino pieces repository for **CondistFL** — a federated learning pipeline for multi-organ medical image segmentation using the ConDistFL framework with NVFlare.

## Pieces

| Piece | Description | Docker Group |
|-------|-------------|--------------|
| **CondistFLDataLoaderPiece** | Downloads NIfTI data from Onedata and prepares it on shared storage | group0 (lightweight) |
| **CondistFLSplitDataPiece** | Splits datasets into cross-validation folds with train/val datalists | group0 (lightweight) |
| **CondistFLTrainerPiece** | Runs NVFlare federated training (4 clients: liver, spleen, pancreas, kidney) | group1 (GPU) |
| **CondistFLVisualizationPiece** | Generates training charts (Dice curves, loss, cross-site heatmap) | group0 (lightweight) |
| **CondistFLInferencePiece** | Runs inference on NIfTI images using trained DynUNet model | group1 (GPU) |

## Pipeline

```
DataLoader → SplitData → Trainer → Visualization
                                  → Inference
```

## Setup

Register this repository in your Domino deployment using the GitHub URL and the version tag from `config.toml`.

## License

MIT — Institute of Informatics Slovak Academy of Sciences (II SAS)
