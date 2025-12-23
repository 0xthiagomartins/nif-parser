"""
Provider-agnostic LLM extraction from .toon OCR output using LiteLLM + Pydantic.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import litellm
from litellm import completion

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from .schemas import RGExtraction


def _get_model() -> str:
    return os.getenv("LLM_MODEL", "openai/gpt-4o-mini")


def _get_api_key() -> str | None:
    return os.getenv("LLM_API_KEY")


def _maybe_enable_debug() -> None:
    if os.getenv("LLM_VERBOSE", "").lower() in {"1", "true", "yes"}:
        litellm.set_verbose = True


def _build_system_prompt(document_type: str) -> str:
    return (
        "Voce extrai dados estruturados de texto OCR.\n"
        "Retorne apenas JSON que siga o schema fornecido.\n"
        "Se um campo estiver ausente, retorne null.\n"
        "Documento: " + document_type
    )


def _model_dump(model: RGExtraction) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()  # pydantic v1 fallback


def extract_from_toon(toon_text: str, document_type: str) -> Dict[str, Any]:
    """
    Extract structured data from .toon text.
    """
    if document_type.lower() != "rg":
        raise ValueError("Only RG schema is available for now")

    litellm.enable_json_schema_validation = True
    _maybe_enable_debug()

    model_name = _get_model()
    api_key = _get_api_key()

    messages = [
        {"role": "system", "content": _build_system_prompt(document_type)},
        {"role": "user", "content": toon_text},
    ]

    try:
        response = completion(
            model=model_name,
            messages=messages,
            response_format=RGExtraction,
            api_key=api_key,
        )
    except litellm.AuthenticationError as exc:
        raise RuntimeError(f"Authentication failed: {exc}") from exc
    except litellm.RateLimitError as exc:
        raise RuntimeError(f"Rate limited: {exc}") from exc
    except litellm.APIError as exc:
        raise RuntimeError(f"API error: {exc}") from exc

    content = response.choices[0].message.content
    parsed = RGExtraction.model_validate_json(content)
    return _model_dump(parsed)


def extract_rg_from_toon(toon_path: str) -> Dict[str, Any]:
    """
    Convenience helper for RG extraction from a .toon file.
    """
    with open(toon_path, "r", encoding="utf-8") as file:
        toon_text = file.read()
    return extract_from_toon(toon_text, "rg")

def extract_rg_from_text(toon_text: str) -> RGExtraction:
    """
    Extract RG data from toon text and return a structured object.
    """
    data = extract_from_toon(toon_text, "rg")
    return RGExtraction.model_validate(data)

