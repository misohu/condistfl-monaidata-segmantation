from pydantic import BaseModel, Field
from typing import Dict, Optional


class InputModel(BaseModel):
    """
    CondistFL Training Input Model
    """
    num_rounds: int = Field(
        default=3,
        description="Number of federated learning rounds",
        json_schema_extra={"from_upstream": "never"}
    )
    steps_per_round: int = Field(
        default=1000,
        description="Training steps per round",
        json_schema_extra={"from_upstream": "never"}
    )
    clients: str = Field(
        default="liver,spleen,pancreas,kidney",
        description="Comma-separated list of client names",
        json_schema_extra={"from_upstream": "never"}
    )
    gpus: str = Field(
        default="0,1,2,3",
        description="Comma-separated GPU IDs to use",
        json_schema_extra={"from_upstream": "never"}
    )
    data_root_kidney: str = Field(
        description="Path to kidney dataset root (from SplitData piece)",
        json_schema_extra={"from_upstream": "always"}
    )
    data_root_liver: str = Field(
        description="Path to liver dataset root (from SplitData piece)",
        json_schema_extra={"from_upstream": "always"}
    )
    data_root_pancreas: str = Field(
        description="Path to pancreas dataset root (from SplitData piece)",
        json_schema_extra={"from_upstream": "always"}
    )
    data_root_spleen: str = Field(
        description="Path to spleen dataset root (from SplitData piece)",
        json_schema_extra={"from_upstream": "always"}
    )
    workspace_dir: str = Field(
        default="/app/workspace",
        description="Directory to save training workspace",
        json_schema_extra={"from_upstream": "never"}
    )


class OutputModel(BaseModel):
    """
    CondistFL Training Output Model
    """
    workspace_dir: str = Field(
        description="Directory containing training results and models"
    )
    best_global_model_path: str = Field(
        description="Path to the best global model checkpoint"
    )
    global_model_path: str = Field(
        description="Path to the final global model checkpoint"
    )
    best_local_models: Dict[str, str] = Field(
        description="Paths to best local models for each client",
        default_factory=dict
    )
    cross_site_validation_results: str = Field(
        description="Path to cross-site validation results YAML file"
    )
    training_complete: bool = Field(
        description="Whether training completed successfully"
    )
    num_rounds_completed: int = Field(
        description="Number of rounds completed"
    )
    validation_metrics: str = Field(
        description="JSON string of validation metrics dict {client_dice: float}",
        default="{}"
    )
    client_metrics: str = Field(
        description="JSON string of per-client TensorBoard scalars {client: {tag: [{step, value}]}}",
        default="{}"
    )
    server_metrics: str = Field(
        description="JSON string of server TensorBoard scalars {tag: [{step, value}]}",
        default="{}"
    )
    cross_val_data: str = Field(
        description="JSON string of cross-site validation results list, or empty string if not available",
        default=""
    )
    message: str = Field(
        description="Status message"
    )
