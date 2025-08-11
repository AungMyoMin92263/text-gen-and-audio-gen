"""Pydantic schemas for API requests and responses."""
from typing import Literal

from pydantic import BaseModel, Field


class StudentRequestModel(BaseModel):
    """Request model for student information."""
    class_name: Literal["ML_in_Prod_1", "ML_in_Prod_2", "Big_Data"]

    stu_name: str = "Mg ba"
    stu_id: int = Field(
        ..., ge=100, le=150, description="Student IDs must between 100 and 150"
    )

    stu_age: int = Field(
        ..., ge=16, le=35, description="Student IDs must between 100 and 150"
    )


class TextRequestModel(BaseModel):
    """Request model for text generation."""
    prompt: str = "What is deep learning"


class TextResponseModel(BaseModel):
    """Response model for text generation results."""
    execution_time: int = 0
    result: str = ""
