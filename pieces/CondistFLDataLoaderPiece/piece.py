import os
import json
import base64
import zipfile
import tempfile
from pathlib import Path
from urllib.parse import quote as urlquote

import requests
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


class CondistFLDataLoaderPiece(BasePiece):
    """
    Downloads CondistFL multi-organ segmentation data from a Onedata provider,
    extracts it to Domino shared storage, and outputs dataset paths for
    downstream pieces.

    The expected zip layout is::

        data_sampled/
            KiTS19/   (datalist.json + *.nii.gz)
            Liver/    …
            Pancreas/ …
            Spleen/   …
    """

    # Dataset folder name -> friendly label
    DATASETS = {
        "KiTS19": "kidney",
        "Liver": "liver",
        "Pancreas": "pancreas",
        "Spleen": "spleen",
    }

    # Onedata REST API version prefix
    _API = "/api/v3/oneprovider"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headers(self, token: str) -> dict:
        return {"X-Auth-Token": token}

    def _resolve_file_id(
        self,
        provider_url: str,
        token: str,
        file_id: str,
        space_name: str,
        file_path: str,
        verify_ssl: bool,
    ) -> str:
        """Return the Onedata File ID – either the one supplied directly,
        or looked up via the path-based endpoint."""
        if file_id:
            self.logger.info(f"Using supplied File ID: {file_id[:20]}…")
            return file_id

        lookup_path = f"{space_name}/{file_path}"
        url = f"{provider_url.rstrip('/')}{self._API}/lookup-file-id/{urlquote(lookup_path, safe='/')}"
        self.logger.info(f"Looking up File ID for /{lookup_path}")

        resp = requests.post(url, headers=self._headers(token), verify=verify_ssl)
        resp.raise_for_status()
        fid = resp.json()["fileId"]
        self.logger.info(f"Resolved File ID: {fid[:20]}…")
        return fid

    def _download_file(
        self,
        provider_url: str,
        token: str,
        file_id: str,
        dest_path: str,
        verify_ssl: bool,
    ) -> None:
        """Stream-download a file from Onedata to *dest_path*."""
        url = f"{provider_url.rstrip('/')}{self._API}/data/{file_id}/content"
        self.logger.info(f"Downloading file {file_id[:20]}… → {dest_path}")

        resp = requests.get(
            url,
            headers=self._headers(token),
            stream=True,
            verify=verify_ssl,
        )
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        chunk_size = 256 * 1024  # 256 KB
        log_interval = 50 * 1024 * 1024  # log every ~50 MB
        next_log_at = log_interval

        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                fh.write(chunk)
                downloaded += len(chunk)
                if total and downloaded >= next_log_at:
                    pct = downloaded * 100 // total
                    self.logger.info(
                        f"Download progress: {downloaded / (1024*1024):.1f} MB / "
                        f"{total / (1024*1024):.1f} MB ({pct}%)"
                    )
                    next_log_at += log_interval

        self.logger.info(
            f"Download complete: {downloaded / (1024*1024):.1f} MB"
        )

    def _extract_and_locate(
        self, zip_path: str, extract_dir: Path
    ) -> Path:
        """
        Extract a zip and return the directory containing the four
        dataset folders.  Handles both flat layout (KiTS19/ at root)
        and nested layout (data_sampled/KiTS19/).
        """
        self.logger.info(f"Extracting {zip_path} → {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        # Determine the actual data root: could be extract_dir itself
        # or a single subdirectory (e.g. data_sampled/)
        for candidate in (extract_dir, *extract_dir.iterdir()):
            if not candidate.is_dir():
                continue
            if all((candidate / folder).is_dir() for folder in self.DATASETS):
                self.logger.info(f"Data root located at {candidate}")
                return candidate

        raise FileNotFoundError(
            f"Could not find all dataset folders ({', '.join(self.DATASETS)}) "
            f"inside extracted archive at {extract_dir}"
        )

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def piece_function(self, input_data: InputModel) -> OutputModel:
        # 1. Resolve Onedata File ID
        file_id = self._resolve_file_id(
            provider_url=input_data.onedata_provider_url,
            token=input_data.onedata_token,
            file_id=input_data.onedata_file_id,
            space_name=input_data.onedata_space_name,
            file_path=input_data.onedata_file_path,
            verify_ssl=input_data.verify_ssl,
        )

        # 2. Download zip to a temp file
        results_dir = Path(getattr(self, "results_path", "/tmp"))
        results_dir.mkdir(parents=True, exist_ok=True)
        zip_path = results_dir / "data_sampled.zip"
        self._download_file(
            provider_url=input_data.onedata_provider_url,
            token=input_data.onedata_token,
            file_id=file_id,
            dest_path=str(zip_path),
            verify_ssl=input_data.verify_ssl,
        )

        # 3. Extract
        data_extract_dir = results_dir / "data"
        data_extract_dir.mkdir(parents=True, exist_ok=True)
        data_root = self._extract_and_locate(str(zip_path), data_extract_dir)

        # 4. Clean up zip to free space
        try:
            zip_path.unlink()
            self.logger.info("Removed zip file to free space")
        except OSError:
            pass

        # 5. Verify datasets
        output_paths: dict[str, str] = {}
        summary_lines: list[str] = []

        for folder, label in self.DATASETS.items():
            src = data_root / folder

            if not src.exists():
                raise FileNotFoundError(f"Dataset folder not found: {src}")

            datalist_file = src / "datalist.json"
            if not datalist_file.exists():
                raise FileNotFoundError(f"datalist.json not found in {src}")

            with open(datalist_file, "r") as f:
                datalist = json.load(f)
            n_samples = len(datalist.get("training", []))

            nifti_files = list(src.glob("*.nii.gz"))

            output_paths[label] = str(src)
            summary_lines.append(
                f"{label} ({folder}): {n_samples} samples, {len(nifti_files)} NIfTI files"
            )
            self.logger.info(
                f"Verified {folder}: {n_samples} samples, {len(nifti_files)} NIfTI files"
            )

        # 6. Display result in Domino UI
        summary_text = (
            "CondistFL Data — downloaded from Onedata\n"
            + "\n".join(summary_lines)
        )
        self.display_result = {
            "file_type": "txt",
            "base64_content": base64.b64encode(
                summary_text.encode("utf-8")
            ).decode("utf-8"),
        }

        return OutputModel(
            kidney_data_path=output_paths["kidney"],
            liver_data_path=output_paths["liver"],
            pancreas_data_path=output_paths["pancreas"],
            spleen_data_path=output_paths["spleen"],
            message=f"Downloaded and verified 4 datasets from Onedata ({data_root})",
        )
