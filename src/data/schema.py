from typing import Literal

from pydantic import BaseModel, Field


class Passage(BaseModel):
    id: str = Field(..., description="Unique identifier for passage")
    title: str = Field(default="", description="Passage or article title")
    text: str = Field(..., description="Passage body text")
    is_supporting: bool = Field(default=False, description="True if marked as supporting fact")


class UnifiedQueryRecord(BaseModel):
    query_id: str = Field(..., description="Deterministic query identifier")
    query: str = Field(..., description="Question string")
    answers: list[str] = Field(default_factory=list, description="Ground truth answer variants")
    dataset: Literal["hotpotqa", "natural_questions"] = Field(..., description="Source dataset")
    passages: list[Passage] = Field(default_factory=list, description="Associated context passages")
