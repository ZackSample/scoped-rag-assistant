"""The gate itself: everything is private unless explicitly allowed through.

Written as deny-by-default on purpose. A new route added later is protected
automatically; the failure mode of forgetting to list something is "it asks me
to log in," not "it was quietly public the whole time."
"""

from collections.abc import Awaitable, Callable

from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.config import get_settings

# Paths reachable without a session. Kept deliberately short.
#   /login            - the login page itself
#   /auth/*           - login/logout/me endpoints
#   /health           - platform probes, which have no session
#   /login-assets/*   - the login page's stylesheet
#   /app/*            - the SPA shell: HTML, JavaScript, CSS. It contains no
#                       machine data - every byte of that comes from the API
#                       routes, which stay gated.
#   /pdf/<token>      - carries its own authorization: POST /pdf-link is gated
#                       and mints a random, single-file, 15-minute token, so the
#                       URL is the credential. Opening a PDF is a new-tab
#                       navigation, which cannot carry an Authorization header.
_PUBLIC_EXACT = {"/login", "/health", "/favicon.ico", "/robots.txt"}
_PUBLIC_PREFIXES = ("/auth/", "/login-assets/", "/app/", "/pdf/")


def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    return any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES)


class RequireLoginMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        settings = get_settings()
        if not settings.auth_required or _is_public(request.url.path):
            return await call_next(request)

        if request.session.get("user_id"):
            return await call_next(request)

        # API callers get a machine-readable 401; humans hitting a page get sent
        # to the login screen. Guessing wrong here means either a browser
        # rendering raw JSON or fetch() following a redirect into HTML.
        accepts_html = "text/html" in request.headers.get("accept", "")
        if accepts_html and request.method == "GET":
            return RedirectResponse(url="/login", status_code=302)
        return JSONResponse({"detail": "Authentication required."}, status_code=401)
