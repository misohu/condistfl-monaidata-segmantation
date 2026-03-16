import os
import json
import shutil
import yaml
import base64
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel


class CondistFLTrainerPiece(BasePiece):
    """
    CondistFL Federated Learning Trainer
    
    This piece trains a federated learning model for multi-organ and tumor 
    segmentation using the CondistFL framework with NVFlare.
    
    The training process:
    1. Configures data paths for each client (kidney, liver, pancreas, spleen)
    2. Launches NVFlare simulator with specified clients and GPUs
    3. Trains for specified number of rounds
    4. Saves best global and local models
    5. Performs cross-site validation
    6. Extracts all TensorBoard metrics for downstream visualization
    """

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_tb_scalars(log_dir: str) -> Dict[str, List[Dict[str, float]]]:
        """
        Read all scalar tags from a TensorBoard event log directory.
        Returns {tag: [{step: int, value: float}, ...]}.
        """
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            ea = EventAccumulator(log_dir)
            ea.Reload()
            result = {}
            for tag in ea.Tags().get("scalars", []):
                events = ea.Scalars(tag)
                result[tag] = [{"step": e.step, "value": e.value} for e in events]
            return result
        except Exception:
            return {}

    @staticmethod
    def _parse_cross_val_yaml(path: Path) -> Optional[List[Dict[str, Any]]]:
        """
        Parse the cross_val_results.yaml written by ReportGenerator.
        Expected format:
            val_results:
              - data_client: "kidney"
                model_owner: "server"
                metrics: {val_meandice_kidney: 0.85, ...}
        Returns list of dicts or None.
        """
        try:
            with open(path, "r") as f:
                data = yaml.safe_load(f)
            if data and isinstance(data, dict) and "val_results" in data:
                return data["val_results"]
            # Fallback: maybe the top-level is already a list
            if isinstance(data, list):
                return data
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def piece_function(self, input_data: InputModel) -> OutputModel:
        """
        Execute the CondistFL federated training
        
        Args:
            input_data: InputModel with training configuration
            
        Returns:
            OutputModel with training results
        """
        # Base directory inside the container where code and jobs were copied
        base_dir = Path("/app")
        
        # Ensure workspace directory exists
        workspace_path = Path(input_data.workspace_dir)
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Starting CondistFL training in workspace: {workspace_path}")
        self.logger.info(f"Clients: {input_data.clients}")
        self.logger.info(f"GPUs: {input_data.gpus}")
        self.logger.info(f"Rounds: {input_data.num_rounds}")
        
        # Update data paths in job configs for each client
        clients = input_data.clients.split(',')
        data_paths = {
            'kidney': input_data.data_root_kidney,
            'liver': input_data.data_root_liver,
            'pancreas': input_data.data_root_pancreas,
            'spleen': input_data.data_root_spleen
        }
        
        jobs_dir = base_dir / "jobs" / "condist"
        
        for client in clients:
            client = client.strip()
            if client in data_paths:
                config_file = jobs_dir / client / "config" / "config_data.json"
                if config_file.exists():
                    self.logger.info(f"Updating data paths for {client}")
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    # Update paths
                    config['data_root'] = data_paths[client]
                    config['data_list'] = f"{data_paths[client]}/datalist.json"
                    
                    with open(config_file, 'w') as f:
                        json.dump(config, f, indent=2)
                else:
                    self.logger.warning(f"Config file not found for {client}: {config_file}")
                    
        # Prepare the training command
        cmd = [
            "nvflare", "simulator",
            "-w", str(workspace_path.absolute()),
            "-c", input_data.clients,
            "-gpu", input_data.gpus,
            str(jobs_dir.absolute())
        ]
        
        # Set environment variable
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{base_dir}/src:{env.get('PYTHONPATH', '')}"
        
        self.logger.info(f"Running training command: {' '.join(cmd)}")
        
        # Execute the training
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                cwd=str(base_dir),
                env=env
            )
            self.logger.info(f"Training stdout: {result.stdout}")
            if result.stderr:
                self.logger.warning(f"Training stderr: {result.stderr}")
            
            training_complete = True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Training failed with error: {e.stderr}")
            training_complete = False
            # Don't raise - we'll still return what we have
        
        # Look for the best global model (corrected path with "server" directory)
        best_model_path = workspace_path / "server" / "simulate_job" / "app_server" / "best_FL_global_model.pt"
        global_model_path = workspace_path / "server" / "simulate_job" / "app_server" / "FL_global_model.pt"
        
        if best_model_path.exists():
            # Copy to shared storage so downstream pieces can access it
            shared_best = os.path.join(self.results_path, "best_FL_global_model.pt")
            shutil.copy2(str(best_model_path), shared_best)
            best_model = shared_best
            self.logger.info(f"Copied best global model to shared storage: {best_model}")
        else:
            best_model = "Not found - training may have failed"
            self.logger.warning("Best global model not found")
            
        if global_model_path.exists():
            shared_global = os.path.join(self.results_path, "FL_global_model.pt")
            shutil.copy2(str(global_model_path), shared_global)
            global_model = shared_global
            self.logger.info(f"Copied final global model to shared storage: {global_model}")
        else:
            global_model = "Not found - training may have failed"
            self.logger.warning("Final global model not found")
        
        # Look for best local models for each client
        best_local_models = {}
        for client in clients:
            client = client.strip()
            local_model_path = workspace_path / client / "models" / "best_model.pt"
            if local_model_path.exists():
                best_local_models[client] = str(local_model_path.absolute())
                self.logger.info(f"Found best local model for {client}: {local_model_path}")
            else:
                self.logger.warning(f"Best local model not found for {client}")
        
        # Look for cross-site validation results
        cross_val_path = workspace_path / "server" / "simulate_job" / "cross_site_val" / "cross_val_results.yaml"
        
        cross_val_data = None
        validation_metrics: Dict[str, float] = {}

        if cross_val_path.exists():
            cross_val_results = str(cross_val_path.absolute())
            self.logger.info(f"Found cross-site validation results: {cross_val_results}")
            cross_val_data = self._parse_cross_val_yaml(cross_val_path)

            # Build validation_metrics summary from cross-val entries
            if cross_val_data:
                for entry in cross_val_data:
                    owner = entry.get("model_owner", "")
                    dc = entry.get("data_client", "")
                    metrics = entry.get("metrics", {})
                    if owner in ("server", "server_best") and "val_meandice" in metrics:
                        validation_metrics[f"{dc}_dice"] = float(metrics["val_meandice"])
        else:
            cross_val_results = "Not found - cross-site validation may not have completed"
            self.logger.warning("Cross-site validation results not found")

        # ----------------------------------------------------------
        # Extract TensorBoard metrics for downstream visualization
        # ----------------------------------------------------------
        client_metrics: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
        for client in clients:
            client = client.strip()
            tb_dir = workspace_path / client / "simulate_job" / f"app_{client}" / "logs"
            if tb_dir.is_dir():
                scalars = self._read_tb_scalars(str(tb_dir))
                if scalars:
                    client_metrics[client] = scalars
                    self.logger.info(
                        f"Extracted TensorBoard metrics for {client}: "
                        f"{list(scalars.keys())}"
                    )

        # Server TensorBoard
        server_tb = workspace_path / "server" / "simulate_job" / "app_server" / "logs"
        server_metrics: Dict[str, List[Dict[str, float]]] = {}
        if server_tb.is_dir():
            server_metrics = self._read_tb_scalars(str(server_tb))
            if server_metrics:
                self.logger.info(f"Extracted server TensorBoard metrics: {list(server_metrics.keys())}")

        # If no cross-val but we have per-client val_meandice from TB, use those
        if not validation_metrics and client_metrics:
            for cname, tags in client_metrics.items():
                if "val_meandice" in tags and tags["val_meandice"]:
                    last = tags["val_meandice"][-1]
                    validation_metrics[f"{cname}_dice"] = float(last["value"])

        # ----------------------------------------------------------
        # Build summary + display_result
        # ----------------------------------------------------------
        num_rounds_completed = input_data.num_rounds if training_complete else 0
        status = "COMPLETED" if training_complete else "FAILED"

        summary_lines = [
            f"CondistFL Training — {status}",
            f"Rounds: {num_rounds_completed}/{input_data.num_rounds}",
            f"Clients: {input_data.clients}",
            "",
        ]

        if validation_metrics:
            summary_lines.append("Validation Dice scores (last round):")
            for k, v in sorted(validation_metrics.items()):
                summary_lines.append(f"  {k}: {v:.4f}")
        else:
            summary_lines.append("No validation metrics available yet.")

        summary_lines.append("")
        summary_lines.append(f"Best global model: {'found' if best_model_path.exists() else 'not found'}")
        summary_lines.append(f"Local models found: {len(best_local_models)}")

        summary_text = "\n".join(summary_lines)
        self.display_result = {
            "file_type": "txt",
            "base64_content": base64.b64encode(summary_text.encode("utf-8")).decode("utf-8"),
        }

        message = "Training completed successfully" if training_complete else "Training failed or incomplete"
        
        return OutputModel(
            workspace_dir=str(workspace_path.absolute()),
            best_global_model_path=best_model,
            global_model_path=global_model,
            best_local_models=best_local_models,
            cross_site_validation_results=cross_val_results,
            training_complete=training_complete,
            num_rounds_completed=num_rounds_completed,
            validation_metrics=json.dumps(validation_metrics),
            client_metrics=json.dumps(client_metrics),
            server_metrics=json.dumps(server_metrics),
            cross_val_data=json.dumps(cross_val_data) if cross_val_data else "",
            message=message
        )

    # Override default container resources for federated learning training
    container_resources = {
        "requests": {
            "cpu": 4000,
            "memory": 8192
        },
        "limits": {
            "cpu": 16000,
            "memory": 32768
        },
        "use_gpu": True,
        "shm_size": 8192  # 8GB shared memory for PyTorch DataLoader multiprocessing
    }
