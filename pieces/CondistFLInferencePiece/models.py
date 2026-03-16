from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class InputModel(BaseModel):
    """
    CondistFL Inference Input Model
    """
    model_config = ConfigDict(protected_namespaces=())

    best_global_model_path: str = Field(
        description="Path to the best global model checkpoint (.pt file) from training",
        json_schema_extra={"from_upstream": "always"}
    )
    image_path: str = Field(
        description="Path to a NIfTI image file (.nii or .nii.gz) for inference",
        json_schema_extra={"from_upstream": "never"}
    )
    use_gpu: bool = Field(
        default=True,
        description="Whether to use GPU for inference (falls back to CPU if unavailable)",
        json_schema_extra={"from_upstream": "never"}
    )
    output_dir: str = Field(
        default="/tmp/condistfl_inference",
        description="Directory to save inference outputs (mask and visualization)",
        json_schema_extra={"from_upstream": "never"}
    )


class OutputModel(BaseModel):
    """
    CondistFL Inference Output Model
    """
    segmentation_mask_path: str = Field(
        description="Path to the predicted segmentation mask NIfTI file"
    )
    visualization_path: str = Field(
        description="Path to the slice visualization PNG file"
    )
    class_names: str = Field(
        description="JSON string mapping class index to organ name"
    )
    message: str = Field(
        description="Status message with inference summary"
    )
