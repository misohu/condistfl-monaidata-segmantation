from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    CondistFL Visualization Input Model
    """
    training_complete: bool = Field(
        description="Whether training completed successfully",
        json_schema_extra={"from_upstream": "always"}
    )
    num_rounds_completed: int = Field(
        description="Number of federated learning rounds completed",
        json_schema_extra={"from_upstream": "always"}
    )
    validation_metrics: str = Field(
        description="JSON string of validation metrics {client_dice: float}",
        default="{}",
        json_schema_extra={"from_upstream": "always"}
    )
    client_metrics: str = Field(
        description="JSON string of per-client TensorBoard scalars",
        default="{}",
        json_schema_extra={"from_upstream": "always"}
    )
    server_metrics: str = Field(
        description="JSON string of server TensorBoard scalars",
        default="{}",
        json_schema_extra={"from_upstream": "always"}
    )
    cross_val_data: str = Field(
        description="JSON string of cross-site validation results, or empty string",
        default="",
        json_schema_extra={"from_upstream": "always"}
    )


class OutputModel(BaseModel):
    """
    CondistFL Visualization Output Model
    """
    charts_dir: str = Field(
        description="Directory containing saved chart PNG files"
    )
    summary: str = Field(
        description="Text summary of training results"
    )
    message: str = Field(
        description="Status message"
    )
