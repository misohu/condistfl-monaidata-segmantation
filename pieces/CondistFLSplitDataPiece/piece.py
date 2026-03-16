import os
import json
import shutil
import base64
import random
from pathlib import Path
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


class CondistFLSplitDataPiece(BasePiece):
    """
    Splits CondistFL multi-organ datasets into k-fold cross-validation
    splits. Creates fold-specific directories with datalist.json files
    containing train/validation partitions, and copies NIfTI files to
    shared storage so the Trainer piece can access them.
    """

    # Map output field name -> (input field name, folder name)
    DATASETS = {
        "kidney": ("kidney_data_path", "KiTS19"),
        "liver": ("liver_data_path", "Liver"),
        "pancreas": ("pancreas_data_path", "Pancreas"),
        "spleen": ("spleen_data_path", "Spleen"),
    }

    def _resolve_data_path(self, upstream_path: str, folder_name: str) -> Path:
        """
        Validate that the upstream path contains a datalist.json.
        Data is expected to come from the DataLoader piece (Onedata download).
        """
        up = Path(upstream_path)
        if (up / "datalist.json").exists():
            return up
        raise FileNotFoundError(
            f"datalist.json not found at upstream path ({up}). "
            f"Ensure the DataLoader piece has downloaded data from Onedata."
        )

    def piece_function(self, input_data: InputModel) -> OutputModel:
        num_folds = input_data.num_folds
        fold_index = input_data.fold_index

        if fold_index < 0 or fold_index >= num_folds:
            raise ValueError(
                f"fold_index ({fold_index}) must be in [0, {num_folds - 1}]"
            )

        results_dir = Path(getattr(self, "results_path", "/tmp"))
        fold_dir = results_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Creating {num_folds}-fold split, using fold {fold_index} for validation"
        )

        output_roots = {}
        summary_lines = []

        for label, (input_field, folder_name) in self.DATASETS.items():
            upstream_path = getattr(input_data, input_field)
            src_path = self._resolve_data_path(upstream_path, folder_name)

            with open(src_path / "datalist.json", "r") as f:
                datalist = json.load(f)

            # Deduplicate: take unique samples from the training list
            samples = datalist.get("training", [])
            seen = set()
            unique_samples = []
            for s in samples:
                key = s["image"]
                if key not in seen:
                    seen.add(key)
                    unique_samples.append(s)

            # Shuffle deterministically
            rng = random.Random(42)
            rng.shuffle(unique_samples)

            # Create fold splits
            fold_size = len(unique_samples) // num_folds
            remainder = len(unique_samples) % num_folds

            folds = []
            start = 0
            for i in range(num_folds):
                end = start + fold_size + (1 if i < remainder else 0)
                folds.append(unique_samples[start:end])
                start = end

            val_samples = folds[fold_index]
            train_samples = []
            for i, fold in enumerate(folds):
                if i != fold_index:
                    train_samples.extend(fold)

            # Create fold output directory
            dataset_dir = fold_dir / folder_name
            dataset_dir.mkdir(parents=True, exist_ok=True)

            # Copy NIfTI files from source to fold directory
            for s in unique_samples:
                for key in ("image", "label"):
                    fname = s[key]
                    src_file = src_path / fname
                    dst_file = dataset_dir / fname
                    if src_file.exists() and not dst_file.exists():
                        shutil.copy2(str(src_file), str(dst_file))

            # Write fold-specific datalist.json
            # "testing" mirrors "validation" – used by NVFlare cross-site validation
            fold_datalist = {
                "training": train_samples,
                "validation": val_samples,
                "testing": val_samples,
            }
            with open(dataset_dir / "datalist.json", "w") as f:
                json.dump(fold_datalist, f, indent=2)

            output_roots[label] = str(dataset_dir)
            summary_lines.append(
                f"{label} ({folder_name}): {len(train_samples)} train, "
                f"{len(val_samples)} val"
            )
            self.logger.info(
                f"{label}: {len(train_samples)} train / {len(val_samples)} val"
            )

        # Display result in Domino UI
        summary = (
            f"Fold {fold_index}/{num_folds} split\n" + "\n".join(summary_lines)
        )
        self.display_result = {
            "file_type": "txt",
            "base64_content": base64.b64encode(
                summary.encode("utf-8")
            ).decode("utf-8"),
        }

        return OutputModel(
            data_root_kidney=output_roots["kidney"],
            data_root_liver=output_roots["liver"],
            data_root_pancreas=output_roots["pancreas"],
            data_root_spleen=output_roots["spleen"],
            fold_index=fold_index,
            num_folds=num_folds,
            message=summary,
        )
