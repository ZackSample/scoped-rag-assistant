"""API endpoints used by the machine selector landing page."""

from fastapi import APIRouter, HTTPException

from app.machines.registry import get_machine_by_number, invalidate_machine_cache, list_machine_records


router = APIRouter(prefix="/machine-registry", tags=["machines"])


@router.get("")
def machine_registry() -> dict:
    """Return friendly machine tiles mapped to their full document folder ids.

    This is the landing page, so it is the natural place to re-read DOCS_ROOT
    and refresh the cached registry - a machine folder added to the share shows
    up as soon as a technician opens the picker.
    """

    invalidate_machine_cache()
    return {"machines": list_machine_records()}


@router.get("/{machine_number}")
def machine_by_number(machine_number: str) -> dict:
    """Resolve a short machine number like 101 before opening its chat."""

    machine = get_machine_by_number(machine_number)
    if not machine:
        raise HTTPException(status_code=404, detail="Machine number was not found.")
    return {"machine": machine}
