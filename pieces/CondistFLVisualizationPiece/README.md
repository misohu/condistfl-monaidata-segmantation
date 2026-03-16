# CondistFL Visualization Piece

Generates visualisation charts from CondistFL federated-learning training results.

## Charts

| Chart | Description |
|-------|-------------|
| **Loss Curves** | Per-client subplots of `loss`, `loss_sup`, `loss_condist` over training steps |
| **Dice Curves** | Overlaid `val_meandice` per client over FL rounds |
| **Per-Organ Dice Bars** | Grouped bar chart of final per-organ Dice scores |
| **Cross-Val Heatmap** | Model owner × data client Dice matrix (when cross-site validation data is available) |

## Inputs

All inputs come from the upstream **CondistFLTrainerPiece**:

- `training_complete` — whether training finished successfully
- `num_rounds_completed` — number of FL rounds
- `validation_metrics` — summary Dice scores
- `client_metrics` — per-client TensorBoard scalar data
- `server_metrics` — server-side TensorBoard scalar data
- `cross_val_data` — parsed cross-site validation YAML (optional)

## Outputs

- `charts_dir` — path to directory containing individual chart PNGs
- `summary` — text summary of results
- `message` — status

A combined dashboard PNG is shown in the Domino UI via `display_result`.
