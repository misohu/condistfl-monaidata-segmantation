from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    CondistFL Data Loader Input Model

    Downloads multi-organ segmentation data from a Onedata provider and
    makes it available on shared storage for downstream pieces.
    """
    onedata_provider_url: str = Field(
        default="https://cloud-sk.data.spice-platform.eu",
        description="URL of the Onedata provider (Oneprovider REST API base)",
        json_schema_extra={"from_upstream": "never"},
    )
    onedata_token: str = Field(
        description="Onedata access token (X-Auth-Token) with read permissions for the target space",
        json_schema_extra={"from_upstream": "never"},
    )
    onedata_file_id: str = Field(
        default="",
        description=(
            "Direct Onedata File ID of data_sampled.zip. "
            "Leave empty to resolve automatically from space_name / file_path."
        ),
        json_schema_extra={"from_upstream": "never"},
    )
    onedata_space_name: str = Field(
        default="ConDistFL",
        description="Onedata space name used for path-based file lookup (ignored when file_id is set)",
        json_schema_extra={"from_upstream": "never"},
    )
    onedata_file_path: str = Field(
        default="data_sampled.zip",
        description="Path to the zip file inside the Onedata space (ignored when file_id is set)",
        json_schema_extra={"from_upstream": "never"},
    )
    verify_ssl: bool = Field(
        default=False,
        description="Whether to verify SSL certificates when contacting the Onedata provider",
        json_schema_extra={"from_upstream": "never"},
    )


class OutputModel(BaseModel):
    """
    CondistFL Data Loader Output Model
    """
    kidney_data_path: str = Field(
        description="Path to kidney (KiTS19) dataset directory"
    )
    liver_data_path: str = Field(
        description="Path to liver dataset directory"
    )
    pancreas_data_path: str = Field(
        description="Path to pancreas dataset directory"
    )
    spleen_data_path: str = Field(
        description="Path to spleen dataset directory"
    )
    message: str = Field(
        description="Status message"
    )
