"""API-key auth for /api/* endpoints.

Set API_KEY in .env to enable. If empty, all endpoints are open (dev mode)
and a warning is logged at startup.
"""
import logging
from fastapi import Header, HTTPException, status

from app.core.runtime_config import runtime_config

log = logging.getLogger("auth")


def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency. Raises 401 if API_KEY is configured and header
    doesn't match. No-op when API_KEY is empty (dev mode)."""
    expected = runtime_config.snapshot().get("api_key", "")
    if not expected:
        return
    if not x_api_key or x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid X-API-Key header",
        )


def log_auth_state() -> None:
    """Called from lifespan startup to surface the auth mode."""
    expected = runtime_config.snapshot().get("api_key", "")
    if expected:
        log.info("API auth enabled (X-API-Key required for /api/*)")
    else:
        log.warning(
            "API_KEY is empty — /api/* endpoints are UNAUTHENTICATED. "
            "Set API_KEY in .env before deploying to a public VPS."
        )
