from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException, status
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.schemas import SignInRequest, SignUpRequest
from app.auth.security import (
    DUMMY_HASH,
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_refresh_token,
    verify_password,
)
from app.config.config import settings
from app.db.models import AuthSession, User


@dataclass
class AuthResult:
    user: User
    access_token: str
    refresh_token: str  # raw; the route puts it in an httpOnly cookie


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _add_session(db: AsyncSession, user_id: int) -> str:
    """Stages a new AuthSession and returns the raw refresh token."""
    raw, token_hash = generate_refresh_token()
    db.add(
        AuthSession(
            user_id=user_id,
            token_hash=token_hash,
            expires_at=_now() + timedelta(days=settings.refresh_token_ttl_days),
        )
    )
    return raw


async def signup(db: AsyncSession, data: SignUpRequest) -> AuthResult:
    email = _normalize_email(data.email)
    conflict = HTTPException(status.HTTP_409_CONFLICT, "Email already registered")

    if await db.scalar(select(User.id).where(User.email == email)):
        raise conflict

    user = User(
        email=email,
        name=data.name.strip(),
        password_hash=await hash_password(data.password),
    )
    db.add(user)
    try:
        await db.flush()  # assigns user.id; catches the unique-email race
    except IntegrityError:
        await db.rollback()
        raise conflict

    refresh_token = _add_session(db, user.id)
    await db.commit()
    return AuthResult(user, create_access_token(user.id), refresh_token)


async def signin(db: AsyncSession, data: SignInRequest) -> AuthResult:
    email = _normalize_email(data.email)
    user = await db.scalar(select(User).where(User.email == email))

    # Always run one verification so response time does not reveal whether the email exists.
    stored_hash = user.password_hash if user and user.password_hash else DUMMY_HASH
    password_ok = await verify_password(data.password, stored_hash)

    if not user or not user.password_hash or not password_ok:
        raise _unauthorized("Invalid email or password")

    refresh_token = _add_session(db, user.id)
    await db.commit()
    return AuthResult(user, create_access_token(user.id), refresh_token)


async def refresh(db: AsyncSession, raw_token: str | None) -> AuthResult:
    if not raw_token:
        raise _unauthorized("Missing refresh token")

    now = _now()
    # FOR UPDATE: two concurrent refreshes with the same token cannot both succeed.
    auth_session = await db.scalar(
        select(AuthSession)
        .where(AuthSession.token_hash == hash_refresh_token(raw_token))
        .with_for_update()
    )
    if auth_session is None:
        raise _unauthorized("Invalid refresh token")

    if auth_session.revoked_at is not None:
        # A rotated/revoked token was replayed -> assume theft, kill every session of the user.
        await db.execute(
            update(AuthSession)
            .where(
                AuthSession.user_id == auth_session.user_id,
                AuthSession.revoked_at.is_(None),
            )
            .values(revoked_at=now)
        )
        await db.commit()
        raise _unauthorized("Refresh token reuse detected, please sign in again")

    if auth_session.expires_at <= now:
        raise _unauthorized("Refresh token expired")

    user = await db.get(User, auth_session.user_id)
    if user is None:
        raise _unauthorized("Invalid refresh token")

    # Rotation: old token dies, a new one is issued.
    auth_session.revoked_at = now
    new_refresh_token = _add_session(db, user.id)
    await db.commit()
    return AuthResult(user, create_access_token(user.id), new_refresh_token)


async def logout(db: AsyncSession, raw_token: str | None) -> None:
    """Idempotent: never fails just because the token is missing/unknown."""
    if not raw_token:
        return
    await db.execute(
        update(AuthSession)
        .where(
            AuthSession.token_hash == hash_refresh_token(raw_token),
            AuthSession.revoked_at.is_(None),
        )
        .values(revoked_at=_now())
    )
    await db.commit()
