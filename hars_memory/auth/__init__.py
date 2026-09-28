"""Authentication and authorization package for HARS long-term memory."""

from __future__ import annotations

from hars_memory.auth.models import LOCAL_SUPERUSER, AccessToken, TokenContext
from hars_memory.auth.policy import (
    AUTH_ENABLED_ENV,
    check_permission,
    is_auth_enabled,
)
from hars_memory.auth.store import (
    DEFAULT_TOKENS_FILE,
    TOKENS_CONFIG_ENV,
    TokenStore,
    get_default_token_store,
)

__all__ = [
    "AUTH_ENABLED_ENV",
    "DEFAULT_TOKENS_FILE",
    "LOCAL_SUPERUSER",
    "TOKENS_CONFIG_ENV",
    "AccessToken",
    "TokenContext",
    "TokenStore",
    "check_permission",
    "get_default_token_store",
    "is_auth_enabled",
]
