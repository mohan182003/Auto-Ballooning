# schemas.py
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field

Point = Tuple[int, int]          # (x, y) pixel position

class Measurement(BaseModel):
    balloon_number: Optional[int] = Field(
        None, description="1-based index assigned after OCR")
    value: str = Field(..., description="Raw value as it appears on the drawing")
    unit: Optional[str] = Field(None, description="mm, in, ° … if any")
    measurement_type: str = Field(..., description="diameter | length | angle | …")
    section: Optional[str] = Field(
        None, description="Detail/section this dimension belongs to, if any")
    bbox: Optional[List[Point]] = Field(
        None, description="4-corner polygon of the text box on the raster page")

class DrawingResponse(BaseModel):
    filename: str
    measurements: List[Measurement]

    class Config:
        # Ignore any future fields coming from build_measurements
        extra = "forbid"          # or "allow" if you prefer to be lenient
