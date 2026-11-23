"""Throttle repeated failed logins.

Two reasons, and the second is the less obvious one:

1. The username/password gate may be the only thing protecting the app, so an
   unthrottled login endpoint is an open invitation to guess passwords.
2. Verifying a password costs 240,000 PBKDF2 iterations by design. That is the
   right cost for a password hash and the wrong thing to let an anonymous
   caller trigger in a loop: a few hundred concurrent attempts would saturate
   the CPU and take the app down for the technicians using it.

Deliberately in-process. The app runs a single instance, and a shared store
would mean adding Redis for a small-scale problem. If it is ever scaled out,
this becomes per-instance rather than useless - state the limitation, don't
pretend it is not there.
"""

import threading
import time
from collections import defaultdict, deque

# Five failures per identity per fifteen minutes. Generous enough that a
# technician mistyping a password on a tablet keyboard is not locked out of
# their shift, tight enough that guessing is hopeless.
MAX_FAILURES = 5
WINDOW_SECONDS = 15 * 60

_failures: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def _prune(key: str, now: float) -> deque:
    attempts = _failures[key]
    while attempts and now - attempts[0] > WINDOW_SECONDS:
        attempts.popleft()
    return attempts


def seconds_until_retry(key: str) -> int:
    """How long this key must wait, or 0 if it may attempt now."""

    now = time.monotonic()
    with _lock:
        attempts = _prune(key, now)
        if len(attempts) < MAX_FAILURES:
            return 0
        return max(1, int(WINDOW_SECONDS - (now - attempts[0])))


def record_failure(key: str) -> None:
    now = time.monotonic()
    with _lock:
        _prune(key, now).append(now)


def clear(key: str) -> None:
    """Forget a key's failures. Called on a successful sign-in, so one bad
    guess before the right password does not count against the technician."""

    with _lock:
        _failures.pop(key, None)


def reset_all() -> None:
    """Test helper - drops all recorded failures."""

    with _lock:
        _failures.clear()
