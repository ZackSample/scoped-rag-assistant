import secrets
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.api.auth_routes import router as auth_router
from app.api.chat_routes import router as chat_router
from app.api.document_routes import router as document_router
from app.api.health_routes import router as health_router
from app.api.machine_routes import router as machine_router
from app.auth.middleware import RequireLoginMiddleware
from app.config import get_settings
from app.db.database import init_db


# This is the single app entrypoint: API routes plus the static technician UI.
@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Make sure the database and schema exist before requests hit the API."""

    init_db()
    yield


def _session_secret() -> str:
    """The signing key for the session cookie.

    Refuses to fall back to a random value when auth is on. A per-boot key
    silently signs everyone out on every restart, which reads to a technician
    as "the app keeps logging me out" rather than as a misconfiguration - so it
    fails loudly at startup instead.
    """

    settings = get_settings()
    if settings.session_secret:
        return settings.session_secret
    if settings.auth_required:
        raise RuntimeError(
            "SESSION_SECRET is not set. Set it to a stable random value:\n"
            '  python -c "import secrets; print(secrets.token_urlsafe(48))"'
        )
    # AUTH_REQUIRED=false is local development only; no session worth keeping.
    return secrets.token_urlsafe(32)


app = FastAPI(title="IMM Troubleshooting Assistant", version="0.2.0", lifespan=lifespan)

# Middleware order matters and is the reverse of what it reads like: Starlette
# runs the LAST-added middleware first, so SessionMiddleware must be added after
# RequireLoginMiddleware for the session to already be decoded by the time the
# login gate inspects it.
app.add_middleware(RequireLoginMiddleware)
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret(),
    session_cookie="imm_session",
    https_only=get_settings().session_https_only,
    same_site="lax",
    max_age=60 * 60 * 12,  # one shift
)


# Registered LAST so it runs FIRST, wrapping everything else - including the
# login gate's own 401/302 responses, which are exactly the ones an anonymous
# crawler would receive.
@app.middleware("http")
async def no_index_headers(request, call_next):
    """Tell search engines not to index or archive anything served here.

    Paired with /robots.txt: robots.txt asks a crawler not to fetch, this tells
    anything that fetched anyway not to publish. Applied to every response, not
    only pages, because a crawler that finds an API URL will index a JSON body
    just as happily as HTML.
    """

    response = await call_next(request)
    response.headers["X-Robots-Tag"] = "noindex, nofollow, noarchive, nosnippet"
    return response


app.include_router(auth_router)
app.include_router(health_router)
app.include_router(machine_router)
app.include_router(chat_router)
app.include_router(document_router)

# Serve the technician app from the same FastAPI process, which keeps local
# testing and single-container deployment simple.
web_dir = Path(__file__).resolve().parents[1] / "web"
if web_dir.exists():

    @app.get("/login", include_in_schema=False)
    def login_page() -> FileResponse:
        """Served as its own route rather than from the /app mount so it stays
        reachable while the login gate is blocking everything else.

        no-store because this page carries the sign-in logic itself; without a
        Cache-Control header a browser may heuristically cache it and quietly
        run an older sign-in against a newer server.
        """

        return FileResponse(web_dir / "login.html", headers={"Cache-Control": "no-store"})

    # Styles for the login page, mounted under a path the gate treats as public
    # so an unauthenticated browser can actually render it.
    app.mount("/login-assets", StaticFiles(directory=web_dir), name="login_assets")

    @app.get("/app/", include_in_schema=False)
    @app.get("/app/index.html", include_in_schema=False)
    def technician_app_index() -> HTMLResponse:
        """Serve index.html with a cache-busting token derived from the assets.

        A hand-written version string in the HTML is never updated, so browsers
        cache app.js indefinitely and a deploy changes nothing anyone can see.
        Deriving the token from the files' modification times means it changes
        exactly when they do, and nobody has to remember anything.
        """

        stamp = max(
            int((web_dir / name).stat().st_mtime) for name in ("app.js", "styles.css") if (web_dir / name).exists()
        )
        html = (web_dir / "index.html").read_text(encoding="utf-8").replace("__ASSET_VERSION__", str(stamp))
        # no-store on the HTML only: it is small, and it is the thing that has to
        # be re-read for a new asset version to ever be noticed.
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    # Registered AFTER the index routes above so those win for "/app/".
    app.mount("/app", StaticFiles(directory=web_dir, html=True), name="technician_app")
