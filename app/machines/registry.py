"""Build the machine registry used by the technician-facing UI.

Technicians should select "101", not a full folder name like
"IMM 101 - ACME (PX 200-50) SN12345". This module bridges those two worlds by
deriving a friendly registry from the document folders under DOCS_ROOT.
"""

from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import threading
import time

from app.rag.machine_resolver import list_machines


logger = logging.getLogger(__name__)


MACHINE_FOLDER_PATTERN = re.compile(r"^IMM\s+(?P<number>\d+)\s*-\s*(?P<details>.+)$", re.IGNORECASE)

# Folders are "IMM <number> - <MAKE> (<model>) <serial>". The serial is
# whatever follows the closing bracket, and matching on that position beats
# matching on the serial's shape: vendors format serials very differently
# (all digits, dashed groups, mixed letters and digits). When the serial
# stayed glued to the model, the model string carried several unrelated number
# groups and the assistant had to guess which one described the machine.
SERIAL_PATTERN = re.compile(r"\)\s*(?P<serial>\S.*?)\s*$")


@dataclass(frozen=True)
class MachineRecord:
    """User-facing machine details mapped to the original machine folder id."""

    machine_id: str
    machine_number: str
    display_name: str
    model: str
    serial: str | None
    folder_name: str

    def to_dict(self) -> dict:
        return {
            "machine_id": self.machine_id,
            "machine_number": self.machine_number,
            "display_name": self.display_name,
            "model": self.model,
            "serial": self.serial,
            "folder_name": self.folder_name,
        }


def parse_machine_folder(folder_name: str) -> MachineRecord | None:
    """Parse a document folder into a simple machine record for the UI."""

    match = MACHINE_FOLDER_PATTERN.match(folder_name)
    if not match:
        return None

    machine_number = match.group("number")
    details = match.group("details").strip()
    serial_match = SERIAL_PATTERN.search(details)
    serial = serial_match.group("serial") if serial_match else None
    # Slice at the start of the serial GROUP, not the match: the match begins at
    # the closing bracket, so cutting there would strip it off the model.
    model = details[: serial_match.start("serial")].strip() if serial_match else details
    return MachineRecord(
        machine_id=folder_name,
        machine_number=machine_number,
        display_name=f"Machine {machine_number}",
        model=model.strip(" -"),
        serial=serial,
        folder_name=folder_name,
    )


def discover_machine_registry() -> list[MachineRecord]:
    """Read machine folders from DOCS_ROOT and convert them into registry rows."""

    records_by_number = {}
    for folder_name in list_machines():
        record = parse_machine_folder(folder_name)
        if record:
            # The UI and session scope are keyed by the short machine number, so
            # duplicate source folders must collapse to one selectable machine.
            # Keep the first sorted folder to make the choice deterministic.
            records_by_number.setdefault(record.machine_number, record)
    return sorted(records_by_number.values(), key=lambda item: int(item.machine_number))


# Machine facts live in a JSON file next to this module rather than a database
# table: the data is small, curated by hand, and reviewable in git. Keys that
# are not machine numbers (e.g. "_readme") are documentation and skipped.
_FACTS_FILENAME = "machine_facts.json"

# Malformed-file warnings are logged once, not per lookup: the registry cache
# refreshes every few minutes, and a broken facts file must degrade to
# "no facts" quietly after the first loud complaint.
_facts_warning_logged = False


def machine_facts_path() -> Path:
    from app.config import get_settings

    override = get_settings().machine_facts_path
    return Path(override) if override else Path(__file__).with_name(_FACTS_FILENAME)


def load_machine_facts() -> dict[str, dict]:
    """Curated identity facts keyed by machine number.

    Facts are identity only - family, tonnage, fitted screw - the things that
    let the answering model tell which row of a multi-machine table belongs to
    the machine in front of it. Answer values (oil capacities, setpoints) are
    deliberately absent: those must come from documents, cited, every time.

    Never raises. Answering a question must not break because someone edited
    this file badly; a bad file means "no facts".
    """

    global _facts_warning_logged
    path = machine_facts_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("machine facts root must be a JSON object")
    except FileNotFoundError:
        return {}
    except Exception:
        if not _facts_warning_logged:
            logger.warning("Could not read machine facts from %s; continuing without them.", path, exc_info=True)
            _facts_warning_logged = True
        return {}

    return {number: facts for number, facts in data.items() if number.isdigit() and isinstance(facts, dict)}


# The registry is derived from folder names, so a lookup needs no database and
# no filesystem walk per request. A short TTL keeps a newly added machine
# folder appearing without a restart while keeping that work off the request
# path - on a network share, listing DOCS_ROOT can take seconds.
_CACHE_TTL_SECONDS = 300
_cache: dict[str, dict] = {}
_cache_expires_at = 0.0
_cache_lock = threading.Lock()


def _cached_machines() -> dict[str, dict]:
    """Machine records keyed by machine_number, refreshed on a TTL."""

    global _cache, _cache_expires_at

    with _cache_lock:
        if _cache_expires_at > time.monotonic():
            return _cache
        facts = load_machine_facts()
        records = {}
        for record in discover_machine_registry():
            entry = record.to_dict()
            # Curated identity facts ride the same cache as the record itself,
            # so a facts edit shows up on the next refresh.
            entry["facts"] = facts.get(record.machine_number, {})
            records[record.machine_number] = entry
        _cache = records
        _cache_expires_at = time.monotonic() + _CACHE_TTL_SECONDS
        return records


def invalidate_machine_cache() -> None:
    """Drop the cached registry so the next lookup re-reads DOCS_ROOT."""

    global _cache_expires_at
    with _cache_lock:
        _cache_expires_at = 0.0


def list_machine_records() -> list[dict]:
    """Every known machine record, numerically sorted.

    NOT named list_machines: this module imports a list_machines() from
    app.rag.machine_resolver that returns raw FOLDER NAMES, and
    discover_machine_registry() calls it. A same-named function here would
    shadow that import and turn discovery into infinite recursion through
    _cached_machines() - which deadlocks on _cache_lock rather than raising.
    """

    return sorted(
        _cached_machines().values(),
        key=lambda record: (
            # Numeric sort where possible so 9 comes before 101.
            int(record["machine_number"]) if record["machine_number"].isdigit() else 10**9,
            record["machine_number"],
        ),
    )


def get_machine_by_number(machine_number: str) -> dict | None:
    """Resolve a short machine number like '101' to the full machine record."""

    return _cached_machines().get(machine_number)


def get_machine_by_id(machine_id: str) -> dict | None:
    """Resolve the original folder/index id to the friendly machine record."""

    for record in _cached_machines().values():
        if record["machine_id"] == machine_id:
            return record
    return None
