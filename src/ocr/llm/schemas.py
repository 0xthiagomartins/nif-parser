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

    # Verso / outros identificadores (nullable)
    data_expedicao: Optional[str]
    via: Optional[str]
    registro_civil: Optional[str]
    dni: Optional[str]

    titulo_eleitor: Optional[str]
    nis_pis_pasep: Optional[str]
    cert_militar: Optional[str]
    cnh: Optional[str]
    cns: Optional[str]

    ctps_numero: Optional[str]
    ctps_serie: Optional[str]
    ctps_uf: Optional[str]
