import os
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


# Consistent colour palette for the four CondistFL clients
CLIENT_COLORS = {
    "kidney": "#E24A90",
    "liver": "#4A90E2",
    "pancreas": "#50C878",
    "spleen": "#F5A623",
}

# Common loss / metric tags we might encounter
LOSS_TAGS = ("loss", "loss_sup", "loss_condist")
DICE_TAG = "val_meandice"


class CondistFLVisualizationPiece(BasePiece):
    """
    Generates visualisation charts from CondistFL federated-learning
    training results.

    Charts produced:
    1. Per-client training loss curves
    2. Per-client validation Dice curves (over FL rounds)
    3. Per-organ final Dice bar chart
    4. Cross-site validation heatmap (if data available)
    """

    # ------------------------------------------------------------------
    # Chart helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_fig(fig, path: Path, dpi: int = 120):
        fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)

    def _chart_loss_curves(
        self, client_metrics: Dict, out_dir: Path
    ) -> Optional[Path]:
        """One subplot per client showing loss, loss_sup, loss_condist."""
        clients_with_loss = [
            c for c in client_metrics
            if any(t in client_metrics[c] for t in LOSS_TAGS)
        ]
        if not clients_with_loss:
            return None

        n = len(clients_with_loss)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

        for idx, client in enumerate(sorted(clients_with_loss)):
            ax = axes[0][idx]
            tags = client_metrics[client]
            for tag in LOSS_TAGS:
                if tag in tags:
                    steps = [p["step"] for p in tags[tag]]
                    vals = [p["value"] for p in tags[tag]]
                    ax.plot(steps, vals, label=tag, linewidth=1.2)
            ax.set_title(client, fontsize=11, fontweight="bold",
                         color=CLIENT_COLORS.get(client, "black"))
            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Training Loss Curves", fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = out_dir / "loss_curves.png"
        self._save_fig(fig, path)
        return path

    def _chart_dice_curves(
        self, client_metrics: Dict, out_dir: Path
    ) -> Optional[Path]:
        """Overlaid val_meandice curves for all clients."""
        clients_with_dice = [
            c for c in client_metrics if DICE_TAG in client_metrics[c]
        ]
        if not clients_with_dice:
            return None

        fig, ax = plt.subplots(figsize=(7, 4))
        for client in sorted(clients_with_dice):
            data = client_metrics[client][DICE_TAG]
            steps = [p["step"] for p in data]
            vals = [p["value"] for p in data]
            colour = CLIENT_COLORS.get(client, None)
            ax.plot(steps, vals, marker="o", label=client, color=colour,
                    linewidth=2, markersize=6)

        ax.set_title("Validation Mean Dice per FL Round",
                      fontsize=13, fontweight="bold")
        ax.set_xlabel("Round")
        ax.set_ylabel("Mean Dice")
        ax.set_ylim(bottom=0)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        path = out_dir / "dice_curves.png"
        self._save_fig(fig, path)
        return path

    def _chart_organ_dice_bars(
        self, client_metrics: Dict, out_dir: Path
    ) -> Optional[Path]:
        """Grouped bar chart of final per-organ Dice for each client."""
        # Collect final per-organ dice values
        organ_data: Dict[str, Dict[str, float]] = {}
        for client, tags in client_metrics.items():
            for tag, points in tags.items():
                if tag.startswith("val_meandice_") and points:
                    organ = tag.replace("val_meandice_", "")
                    organ_data.setdefault(client, {})[organ] = points[-1]["value"]

        if not organ_data:
            return None

        clients = sorted(organ_data.keys())
        all_organs = sorted({o for d in organ_data.values() for o in d})
        if not all_organs:
            return None

        import numpy as np
        x = np.arange(len(all_organs))
        width = 0.8 / max(len(clients), 1)

        fig, ax = plt.subplots(figsize=(max(7, len(all_organs) * 2), 5))
        for i, client in enumerate(clients):
            vals = [organ_data[client].get(o, 0) for o in all_organs]
            colour = CLIENT_COLORS.get(client, None)
            bars = ax.bar(x + i * width, vals, width, label=client, color=colour)
            # Value labels
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_title("Per-Organ Dice Scores (Final Round)",
                      fontsize=13, fontweight="bold")
        ax.set_xlabel("Organ")
        ax.set_ylabel("Dice")
        ax.set_xticks(x + width * (len(clients) - 1) / 2)
        ax.set_xticklabels([o.replace("_", " ").title() for o in all_organs])
        ax.set_ylim(0, min(1.05, ax.get_ylim()[1] * 1.15))
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        path = out_dir / "organ_dice_bars.png"
        self._save_fig(fig, path)
        return path

    def _chart_crossval_heatmap(
        self, cross_val_data: List[Dict[str, Any]], out_dir: Path
    ) -> Optional[Path]:
        """Heatmap of model_owner × data_client → mean Dice."""
        if not cross_val_data:
            return None

        import numpy as np

        entries = [e for e in cross_val_data if "val_meandice" in e.get("metrics", {})]
        if not entries:
            return None

        owners = sorted({e["model_owner"] for e in entries})
        data_clients = sorted({e["data_client"] for e in entries})

        matrix = np.full((len(owners), len(data_clients)), float("nan"))
        for e in entries:
            r = owners.index(e["model_owner"])
            c = data_clients.index(e["data_client"])
            matrix[r, c] = e["metrics"]["val_meandice"]

        fig, ax = plt.subplots(figsize=(max(6, len(data_clients) * 1.8),
                                        max(4, len(owners) * 1.2)))
        im = ax.imshow(matrix, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(data_clients)))
        ax.set_xticklabels(data_clients, rotation=30, ha="right")
        ax.set_yticks(range(len(owners)))
        ax.set_yticklabels(owners)
        ax.set_xlabel("Data Client")
        ax.set_ylabel("Model Owner")
        ax.set_title("Cross-Site Validation — Mean Dice",
                      fontsize=13, fontweight="bold")

        # Annotate cells
        for r in range(len(owners)):
            for c in range(len(data_clients)):
                v = matrix[r, c]
                if not (v != v):  # not NaN
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=10, fontweight="bold",
                            color="white" if v > 0.5 else "black")

        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        path = out_dir / "crossval_heatmap.png"
        self._save_fig(fig, path)
        return path

    # ------------------------------------------------------------------
    # Combined dashboard
    # ------------------------------------------------------------------

    def _build_dashboard(self, chart_paths: List[Path], out_dir: Path) -> Path:
        """
        Stitch individual charts vertically into one dashboard PNG used
        for display_result.
        """
        from PIL import Image

        images = []
        for p in chart_paths:
            if p and p.exists():
                images.append(Image.open(p))

        if not images:
            # Fallback: create a simple "no charts" image
            fig, ax = plt.subplots(figsize=(6, 2))
            ax.text(0.5, 0.5, "No training metrics to visualise",
                    ha="center", va="center", fontsize=14)
            ax.axis("off")
            path = out_dir / "dashboard.png"
            self._save_fig(fig, path)
            return path

        total_w = max(im.width for im in images)
        total_h = sum(im.height for im in images)
        dashboard = Image.new("RGB", (total_w, total_h), "white")
        y = 0
        for im in images:
            dashboard.paste(im, (0, y))
            y += im.height

        path = out_dir / "dashboard.png"
        dashboard.save(str(path))
        return path

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def piece_function(self, input_data: InputModel) -> OutputModel:
        import json as _json

        results_dir = Path(getattr(self, "results_path", "/tmp"))
        charts_dir = results_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        # Deserialize JSON strings from upstream
        client_metrics = _json.loads(input_data.client_metrics) if input_data.client_metrics else {}
        server_metrics = _json.loads(input_data.server_metrics) if input_data.server_metrics else {}
        validation_metrics = _json.loads(input_data.validation_metrics) if input_data.validation_metrics else {}
        cross_val_data = _json.loads(input_data.cross_val_data) if input_data.cross_val_data else None

        self.logger.info(
            f"Generating visualisations — training_complete={input_data.training_complete}, "
            f"rounds={input_data.num_rounds_completed}"
        )

        chart_paths: List[Path] = []

        # 1. Loss curves
        p = self._chart_loss_curves(client_metrics, charts_dir)
        if p:
            chart_paths.append(p)
            self.logger.info("Generated loss curves chart")

        # 2. Dice convergence curves
        p = self._chart_dice_curves(client_metrics, charts_dir)
        if p:
            chart_paths.append(p)
            self.logger.info("Generated Dice curves chart")

        # 3. Per-organ Dice bar chart
        p = self._chart_organ_dice_bars(client_metrics, charts_dir)
        if p:
            chart_paths.append(p)
            self.logger.info("Generated per-organ Dice bar chart")

        # 4. Cross-site validation heatmap
        if cross_val_data:
            p = self._chart_crossval_heatmap(cross_val_data, charts_dir)
            if p:
                chart_paths.append(p)
                self.logger.info("Generated cross-validation heatmap")

        # Build combined dashboard
        dashboard_path = self._build_dashboard(chart_paths, charts_dir)

        # Encode for Domino UI display
        with open(dashboard_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        self.display_result = {
            "file_type": "png",
            "base64_content": img_b64,
        }

        # Text summary
        summary_lines = [
            f"Training complete: {input_data.training_complete}",
            f"Rounds completed: {input_data.num_rounds_completed}",
            f"Charts generated: {len(chart_paths)}",
        ]
        if validation_metrics:
            summary_lines.append("Validation Dice:")
            for k, v in sorted(validation_metrics.items()):
                summary_lines.append(f"  {k}: {v:.4f}")

        summary = "\n".join(summary_lines)

        return OutputModel(
            charts_dir=str(charts_dir),
            summary=summary,
            message=f"Generated {len(chart_paths)} chart(s)",
        )
