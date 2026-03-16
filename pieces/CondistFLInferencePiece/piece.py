import os
import json
import traceback

import numpy as np
import nibabel as nib
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from monai.networks.nets import DynUNet
from monai.inferers import SlidingWindowInferer
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    LoadImage,
    Orientation,
    Spacing,
)

from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


# ── Constants ────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "background",
    1: "liver",
    2: "liver_tumor",
    3: "spleen",
    4: "pancreas",
    5: "pancreas_tumor",
    6: "kidney",
    7: "kidney_tumor",
}

# Colours for each foreground class (background is transparent)
CLASS_COLORS = {
    1: "#4A90E2",   # liver – blue
    2: "#E24A90",   # liver_tumor – pink
    3: "#50C878",   # spleen – green
    4: "#F5A623",   # pancreas – orange
    5: "#BD10E0",   # pancreas_tumor – purple
    6: "#D0021B",   # kidney – red
    7: "#F8E71C",   # kidney_tumor – yellow
}

NUM_CLASSES = 8

# DynUNet architecture (identical to CondistFL config_task.json)
DYNUNET_ARGS = dict(
    spatial_dims=3,
    in_channels=1,
    out_channels=NUM_CLASSES,
    kernel_size=[[3, 3, 1], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
    strides=[[1, 1, 1], [2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    upsample_kernel_size=[[2, 2, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    deep_supervision=True,
    deep_supr_num=3,
)

ROI_SIZE = [224, 224, 64]


# ── Custom normalisation (mirrors src/data/normalize.py) ────────────
class NormalizeIntensityRange:
    """Clip to [a_min, a_max] then shift/scale: (x - subtrahend) / divisor."""

    def __init__(self, a_min: float, a_max: float, subtrahend: float, divisor: float):
        self.a_min = a_min
        self.a_max = a_max
        self.subtrahend = subtrahend
        self.divisor = divisor

    def __call__(self, img):
        img = np.clip(img, self.a_min, self.a_max)
        img = (img - self.subtrahend) / self.divisor
        return img.astype(np.float32)


# ── Piece ────────────────────────────────────────────────────────────
class CondistFLInferencePiece(BasePiece):
    """
    Runs inference on a single NIfTI image using a trained CondistFL
    global model checkpoint.

    Outputs:
      • Segmentation mask saved as NIfTI (.nii.gz)
      • Slice visualisation saved as PNG
    """

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pre_transforms():
        """
        Image-only preprocessing that mirrors the CondistFL validation /
        inference transforms (without dict keys).
        """
        return Compose([
            LoadImage(image_only=True),
            EnsureChannelFirst(),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=[1.44423774, 1.44423774, 2.87368553], mode="bilinear"),
            NormalizeIntensityRange(a_min=-54.0, a_max=258.0, subtrahend=100.0, divisor=50.0),
            EnsureType(data_type="tensor"),
        ])

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    @staticmethod
    def _build_model(device: torch.device) -> DynUNet:
        model = DynUNet(**DYNUNET_ARGS)
        model = model.to(device)
        model.eval()
        return model

    @staticmethod
    def _load_checkpoint(model: DynUNet, ckpt_path: str, device: torch.device) -> DynUNet:
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict)
        return model

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    @staticmethod
    def _pick_representative_slice(mask_vol: np.ndarray) -> int:
        """Return axial slice with the most foreground voxels."""
        fg = (mask_vol > 0).astype(np.int32)
        counts = fg.sum(axis=(0, 1))  # sum over H, W → per-slice count
        if counts.max() == 0:
            return mask_vol.shape[2] // 2
        return int(np.argmax(counts))

    def _create_visualization(
        self,
        image_vol: np.ndarray,
        mask_vol: np.ndarray,
        out_path: str,
    ) -> None:
        """Save a multi-panel PNG: image | overlay | mask."""
        slice_idx = self._pick_representative_slice(mask_vol)
        img_slice = image_vol[:, :, slice_idx]
        msk_slice = mask_vol[:, :, slice_idx]

        # Build RGBA overlay from mask
        cmap_list = ["#000000"] + [CLASS_COLORS.get(i, "#FFFFFF") for i in range(1, NUM_CLASSES)]
        cmap = mcolors.ListedColormap(cmap_list)
        norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, NUM_CLASSES), ncolors=NUM_CLASSES)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Panel 1: raw image
        axes[0].imshow(img_slice.T, cmap="gray", origin="lower")
        axes[0].set_title("Input image", fontsize=13)
        axes[0].axis("off")

        # Panel 2: image + overlay
        axes[1].imshow(img_slice.T, cmap="gray", origin="lower")
        overlay = np.ma.masked_where(msk_slice == 0, msk_slice)
        axes[1].imshow(overlay.T, cmap=cmap, norm=norm, alpha=0.55, origin="lower")
        axes[1].set_title("Prediction overlay", fontsize=13)
        axes[1].axis("off")

        # Panel 3: mask alone
        im = axes[2].imshow(msk_slice.T, cmap=cmap, norm=norm, origin="lower")
        axes[2].set_title("Segmentation mask", fontsize=13)
        axes[2].axis("off")

        # Colour-bar legend
        cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, ticks=range(NUM_CLASSES))
        cbar.ax.set_yticklabels([CLASS_NAMES[i] for i in range(NUM_CLASSES)], fontsize=8)

        # Per-class voxel counts
        unique, counts = np.unique(mask_vol[mask_vol > 0], return_counts=True)
        parts = []
        for cls_id, cnt in zip(unique, counts):
            parts.append(f"{CLASS_NAMES.get(int(cls_id), '?')}: {cnt:,} voxels")
        stats_text = "  |  ".join(parts) if parts else "No foreground detected"

        fig.suptitle(
            f"CondistFL Inference  —  axial slice {slice_idx}\n{stats_text}",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.logger.info(f"Visualization saved: {out_path}")

    # ------------------------------------------------------------------
    # Main entry-point
    # ------------------------------------------------------------------

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info("=" * 70)
            self.logger.info("CondistFLInferencePiece – starting")
            self.logger.info("=" * 70)

            # ── Validate inputs ──────────────────────────────────────
            ckpt_path = input_data.best_global_model_path
            image_path = input_data.image_path
            output_dir = input_data.output_dir

            if not os.path.isfile(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"Checkpoint : {ckpt_path}")
            self.logger.info(f"Image      : {image_path}")
            self.logger.info(f"Output dir : {output_dir}")

            # ── Device ───────────────────────────────────────────────
            if input_data.use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Device: GPU ({torch.cuda.get_device_name(0)})")
            else:
                device = torch.device("cpu")
                self.logger.info("Device: CPU")

            # ── Build model & load weights ───────────────────────────
            self.logger.info("Building DynUNet model …")
            model = self._build_model(device)
            model = self._load_checkpoint(model, ckpt_path, device)
            self.logger.info("Model weights loaded successfully")

            # ── Preprocess image ─────────────────────────────────────
            self.logger.info("Preprocessing image …")
            pre_transforms = self._build_pre_transforms()
            img_tensor = pre_transforms(image_path)           # (C, H, W, D)
            if img_tensor.ndim == 3:
                img_tensor = img_tensor.unsqueeze(0)          # add channel
            img_tensor = img_tensor.unsqueeze(0).to(device)   # (1, C, H, W, D)
            self.logger.info(f"Image tensor shape: {tuple(img_tensor.shape)}")

            # ── Keep a copy for the visualisation ────────────────────
            # Reload the raw image (without normalisation) for display.
            raw_nii = nib.load(image_path)
            affine = raw_nii.affine
            raw_data = raw_nii.get_fdata()

            # ── Inference ────────────────────────────────────────────
            self.logger.info("Running sliding-window inference …")
            inferer = SlidingWindowInferer(
                roi_size=ROI_SIZE,
                sw_batch_size=1,
                mode="gaussian",
                overlap=0.5,
            )

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = inferer(img_tensor, model)
                # DynUNet with deep supervision may return a list – take first
                if isinstance(logits, (list, tuple)):
                    logits = logits[0]
                pred = torch.argmax(logits, dim=1).squeeze(0)  # (H, W, D)

            mask_np = pred.cpu().numpy().astype(np.uint8)
            self.logger.info(f"Prediction shape: {mask_np.shape}")

            # Log per-class counts
            for cls_id in range(1, NUM_CLASSES):
                count = int((mask_np == cls_id).sum())
                if count > 0:
                    self.logger.info(f"  {CLASS_NAMES[cls_id]}: {count:,} voxels")

            # ── Save segmentation mask as NIfTI ──────────────────────
            basename = os.path.splitext(os.path.splitext(os.path.basename(image_path))[0])[0]
            mask_filename = f"{basename}_seg.nii.gz"
            mask_path = os.path.join(output_dir, mask_filename)

            mask_nii = nib.Nifti1Image(mask_np, affine)
            nib.save(mask_nii, mask_path)
            self.logger.info(f"Segmentation mask saved: {mask_path}")

            # ── Save visualisation ───────────────────────────────────
            viz_filename = f"{basename}_inference.png"
            viz_path = os.path.join(output_dir, viz_filename)

            # Resample raw image to the preprocessed grid for consistent
            # overlay – but use the preprocessed tensor directly instead.
            img_for_viz = img_tensor.squeeze().cpu().numpy()  # (H, W, D)
            self._create_visualization(img_for_viz, mask_np, viz_path)

            # ── Display result ───────────────────────────────────────
            self.display_result = {
                "file_type": "png",
                "file_path": viz_path,
            }

            # ── Summary ─────────────────────────────────────────────
            unique_classes = [int(c) for c in np.unique(mask_np) if c > 0]
            detected = [CLASS_NAMES[c] for c in unique_classes]
            msg = (
                f"Inference complete. "
                f"Detected {len(detected)} class(es): {', '.join(detected) if detected else 'none'}."
            )

            self.logger.info("=" * 70)
            self.logger.info(msg)
            self.logger.info("=" * 70)

            return OutputModel(
                segmentation_mask_path=mask_path,
                visualization_path=viz_path,
                class_names=json.dumps(CLASS_NAMES),
                message=msg,
            )

        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            self.logger.error(traceback.format_exc())
            raise
