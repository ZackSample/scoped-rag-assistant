"""Small service-status endpoints."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse, RedirectResponse


router = APIRouter()


@router.get("/health")
def health() -> dict:
    """Return a lightweight status response for tests and deployment probes."""

    return {"status": "ok"}


@router.get("/robots.txt", include_in_schema=False)
def robots() -> PlainTextResponse:
    """Ask crawlers to stay out entirely.

    This is a request, not a control: well-behaved crawlers honour it and
    nothing else does. The X-Robots-Tag header in app/main.py covers the same
    ground for anything that fetches a page directly, and neither is a
    substitute for restricting who can reach the site at all.
    """

    return PlainTextResponse("User-agent: *\nDisallow: /\n")


@router.get("/")
def root() -> RedirectResponse:
    """Send users who open the base URL to the technician app."""

    return RedirectResponse(url="/app/")
