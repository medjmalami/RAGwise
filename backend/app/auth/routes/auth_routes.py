from fastapi import APIRouter, Cookie, Depends, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.controllers import auth_controller
from app.auth.dependencies import get_current_user
from app.auth.schemas import AuthResponse, SignInRequest, SignUpRequest, UserOut
from app.config.config import settings
from app.db.db import get_session
from app.db.models import User

router = APIRouter(prefix="/auth", tags=["auth"])

REFRESH_COOKIE = "refresh_token"
COOKIE_PATH = "/auth"  # the browser only sends the cookie to auth endpoints


def _set_refresh_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        key=REFRESH_COOKIE,
        value=token,
        max_age=settings.refresh_token_ttl_days * 24 * 3600,
        httponly=True,
        secure=settings.cookie_secure,  # True in production (HTTPS)
        samesite="lax",
        path=COOKIE_PATH,
    )


def _clear_refresh_cookie(response: Response) -> None:
    response.delete_cookie(
        key=REFRESH_COOKIE,
        path=COOKIE_PATH,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="lax",
    )


def _to_response(result: auth_controller.AuthResult) -> AuthResponse:
    return AuthResponse(
        access_token=result.access_token,
        expires_in=settings.access_token_ttl_minutes * 60,
        user=UserOut.model_validate(result.user),
    )


@router.post(
    "/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED
)
async def signup(
    body: SignUpRequest,
    response: Response,
    db: AsyncSession = Depends(get_session),
):
    result = await auth_controller.signup(db, body)
    _set_refresh_cookie(response, result.refresh_token)
    return _to_response(result)


@router.post("/signin", response_model=AuthResponse)
async def signin(
    body: SignInRequest,
    response: Response,
    db: AsyncSession = Depends(get_session),
):
    result = await auth_controller.signin(db, body)
    _set_refresh_cookie(response, result.refresh_token)
    return _to_response(result)


@router.post("/refresh", response_model=AuthResponse)
async def refresh(
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
    db: AsyncSession = Depends(get_session),
):
    result = await auth_controller.refresh(db, refresh_token)
    _set_refresh_cookie(response, result.refresh_token)
    return _to_response(result)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    response: Response,
    refresh_token: str | None = Cookie(default=None, alias=REFRESH_COOKIE),
    db: AsyncSession = Depends(get_session),
):
    await auth_controller.logout(db, refresh_token)
    _clear_refresh_cookie(response)


@router.get("/me", response_model=UserOut)
async def me(user: User = Depends(get_current_user)):
    return user
