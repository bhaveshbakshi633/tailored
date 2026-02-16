"""
API key authentication middleware.

Production deployment should replace this with a proper auth system
(JWT / OAuth2).  This provides a minimal gate for the MVP.
"""

from __future__ import annotations

import os
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader

from app.config import config

_api_key_header = APIKeyHeader(name=config.api_key_header, auto_error=False)

# In production, load from a secrets manager.
# For development, use an env var or a default.
_VALID_KEYS: set[str] = set()


def _load_keys() -> None:
    """Load API keys from the GEOMCALC_API_KEYS environment variable."""
    raw = os.environ.get("GEOMCALC_API_KEYS", "")
    if raw:
        _VALID_KEYS.update(k.strip() for k in raw.split(",") if k.strip())


_load_keys()


async def require_api_key(
    api_key: Annotated[str | None, Security(_api_key_header)],
) -> str:
    """Dependency: reject requests without a valid API key."""
    if not _VALID_KEYS:
        # No keys configured → auth disabled (development mode)
        return "dev"

    if api_key is None or not secrets.compare_digest(api_key, api_key):
        raise HTTPException(401, "Invalid or missing API key")

    if api_key not in _VALID_KEYS:
        raise HTTPException(401, "Invalid or missing API key")

    return api_key
