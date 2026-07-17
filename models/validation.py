"""Model availability checks against provider model lists."""

from typing import Dict, List, Set

from .base import APIClient


class ModelListError(RuntimeError):
    """Raised when a provider model list cannot be retrieved."""


class ModelNotFoundError(RuntimeError):
    """Raised when a requested model is absent from the provider model list."""


def model_identifiers(model_entry: Dict) -> Set[str]:
    """Return known identifier strings for one list_models() entry."""
    ids: Set[str] = set()
    for key in ("id", "name"):
        value = model_entry.get(key)
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                ids.add(stripped)
    return ids


def _ollama_names_equivalent(wanted: str, listed: str) -> bool:
    if wanted == listed:
        return True
    wanted_base, _, wanted_tag = wanted.partition(":")
    listed_base, _, listed_tag = listed.partition(":")
    if wanted_base != listed_base:
        return False
    if not wanted_tag or not listed_tag:
        return True
    return wanted_tag == listed_tag


def is_model_listed(model_name: str, models: List[Dict], *, provider: str = "") -> bool:
    """Return True when model_name matches a provider model list entry."""
    wanted = model_name.strip()
    if not wanted:
        return False

    provider_key = provider.lower()
    for entry in models:
        for ident in model_identifiers(entry):
            if ident == wanted:
                return True
            if provider_key == "ollama" and _ollama_names_equivalent(wanted, ident):
                return True
    return False


def _format_available_models(models: List[Dict], *, limit: int = 20) -> str:
    available = sorted({ident for entry in models for ident in model_identifiers(entry)})
    if not available:
        return "(none)"
    shown = available[:limit]
    hint = ", ".join(shown)
    if len(available) > limit:
        hint += ", ..."
    return hint


def validate_model_available(
    client: APIClient,
    model_name: str,
    *,
    provider: str,
    role: str = "target",
) -> None:
    """Raise when model_name is not listed by the provider."""
    try:
        models = client.list_models()
    except RuntimeError as e:
        raise ModelListError(
            f"Cannot verify {role} model {model_name!r}: {e}"
        ) from e

    if not models:
        raise ModelListError(
            f"Cannot verify {role} model {model_name!r}: provider returned an empty model list."
        )

    if is_model_listed(model_name, models, provider=provider):
        return

    available = _format_available_models(models)
    raise ModelNotFoundError(
        f"{role.capitalize()} model {model_name!r} not found at {client.base_url}. "
        f"Available: {available}"
    )
