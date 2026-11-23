"""Login/logout endpoints for the app's own username+password gate.

Besides access control, its other job is per-user attribution: knowing which
technician asked a question when devices on the floor are shared.
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from app.auth import rate_limit
from app.auth.users import authenticate

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1)
    password: str = Field(min_length=1)


def _rate_limit_keys(payload: LoginRequest, request: Request) -> list[str]:
    """Throttle on username AND source address.

    Username alone lets one attacker lock out a real technician by guessing at
    their account; address alone is defeated by rotating usernames. Both are
    counted, and either one tripping is enough to refuse.
    """

    client = request.client.host if request.client else "unknown"
    return [f"user:{payload.username.strip().lower()}", f"ip:{client}"]


@router.post("/login")
def login(payload: LoginRequest, request: Request) -> dict:
    keys = _rate_limit_keys(payload, request)
    for key in keys:
        wait = rate_limit.seconds_until_retry(key)
        if wait:
            # Refused before authenticate() is called, so a throttled attempt
            # never reaches the deliberately expensive password hash.
            return {
                "ok": False,
                "error": f"Too many failed sign-ins. Try again in {wait // 60 + 1} minute(s).",
                "retry_after_seconds": wait,
            }

    user = authenticate(payload.username, payload.password)
    if user is None:
        for key in keys:
            rate_limit.record_failure(key)
        # One generic message for every failure mode - no username enumeration.
        return {"ok": False, "error": "Invalid username or password."}

    for key in keys:
        rate_limit.clear(key)
    request.session["user_id"] = user.user_id
    request.session["username"] = user.username
    request.session["display_name"] = user.display_name
    return {"ok": True, "user": {"username": user.username, "display_name": user.display_name}}


@router.post("/logout")
def logout(request: Request) -> dict:
    request.session.clear()
    return {"ok": True}


@router.get("/me")
def me(request: Request) -> dict:
    """Who is signed in on this session. Used by the UI to render the user's name."""

    username = request.session.get("username")
    if not username or not request.session.get("user_id"):
        return {"authenticated": False}
    return {
        "authenticated": True,
        "user": {"username": username, "display_name": request.session.get("display_name") or username},
    }
