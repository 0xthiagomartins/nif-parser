"""
Pydantic models for structured extraction.
"""

from typing import Optional
from pydantic import BaseModel


class RGExtraction(BaseModel):
    document_type: str
    full_name: Optional[str]
    rg_number: Optional[str]
    issuer: Optional[str]
    birth_date: Optional[str]
    naturalidade: Optional[str]
    state: Optional[str]
    filiacao_pai: Optional[str]
    filiacao_mae: Optional[str]
    observacoes: Optional[str]
    confidence: Optional[float]
