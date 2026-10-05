import asyncio
import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

from app.config.config import settings

_ph = PasswordHasher()
# Used to burn the same CPU time when the email does not exist (anti user-enumeration).
DUMMY_HASH = _ph.hash("dummy-password-for-timing-equalisation")


# ---------- passwords (argon2 is CPU-bound -> keep it off the event loop) ----------
async def hash_password(password: str) -> str:
    return await asyncio.to_thread(_ph.hash, password)


async def verify_password(password: str, password_hash: str) -> bool:
    def _verify() -> bool:
        try:
            return _ph.verify(password_hash, password)
        except (VerifyMismatchError, InvalidHashError):
            return False

    return await asyncio.to_thread(_verify)


# ---------- access token (short-lived JWT) ----------
def create_access_token(user_id: int) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_ttl_minutes),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> int:
    """Returns the user id. Raises jwt.PyJWTError if invalid/expired."""
    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[settings.jwt_algorithm],
        options={"require": ["exp", "sub"]},
    )
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("wrong token type")
    return int(payload["sub"])


# ---------- refresh token (opaque, only its SHA-256 is stored) ----------
def hash_refresh_token(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def generate_refresh_token() -> tuple[str, str]:
    """Returns (raw_token, token_hash)."""
    raw = secrets.token_urlsafe(48)
    return raw, hash_refresh_token(raw)
