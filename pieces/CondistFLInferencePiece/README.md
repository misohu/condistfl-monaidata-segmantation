# CondistFL Inference Piece

Runs inference on a single NIfTI CT image using a trained CondistFL federated‑learning global model checkpoint.

## Inputs

| Parameter | Type | Default | Upstream | Description |
|-----------|------|---------|----------|-------------|
| `best_global_model_path` | `str` | — | always | Path to the best global model `.pt` checkpoint (from Trainer piece) |
| `image_path` | `str` | — | never | Path to a NIfTI image (`.nii` / `.nii.gz`) |
| `use_gpu` | `bool` | `true` | never | Use GPU if available |
| `output_dir` | `str` | `/tmp/condistfl_inference` | never | Directory for output files |

## Outputs

| Field | Type | Description |
|-------|------|-------------|
| `segmentation_mask_path` | `str` | Path to the predicted segmentation mask (`.nii.gz`) |
| `visualization_path` | `str` | Path to the slice visualisation PNG |
| `class_names` | `str` | JSON mapping of class index → organ name |
| `message` | `str` | Human‑readable summary |

## Segmentation classes

| Index | Class |
|-------|-------|
| 0 | background |
| 1 | liver |
| 2 | liver_tumor |
| 3 | spleen |
| 4 | pancreas |
| 5 | pancreas_tumor |
| 6 | kidney |
| 7 | kidney_tumor |

## Pipeline position

This piece receives `best_global_model_path` from the **CondistFL Trainer** piece and runs in parallel with the **CondistFL Visualization** piece.

```
DataLoader → SplitData → Trainer ─┬─→ Visualization
                                   └─→ Inference
```
