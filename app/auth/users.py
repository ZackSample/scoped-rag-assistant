"""The app's user store.

Users are created with `scripts/create_user.py`, not self-service
registration - there is deliberately no signup path.
"""

from dataclasses import dataclass

from app.auth.passwords import verify_password
from app.db.database import connect


@dataclass
class AppUser:
    user_id: int
    username: str
    display_name: str
    is_active: bool


def authenticate(username: str, password: str) -> AppUser | None:
    """Return the user when the credentials are valid and active, else None.

    Callers get a single opaque None for "no such user", "wrong password", and
    "deactivated" so the login endpoint can't be used to enumerate usernames.
    """

    with connect() as conn:
        row = conn.execute(
            "SELECT user_id, username, display_name, password_hash, is_active FROM app_users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
        if row is None or not row["is_active"]:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        conn.execute("UPDATE app_users SET last_login_at = CURRENT_TIMESTAMP WHERE user_id = ?", (row["user_id"],))
        return AppUser(
            user_id=row["user_id"],
            username=row["username"],
            display_name=row["display_name"],
            is_active=bool(row["is_active"]),
        )


def upsert_user(username: str, display_name: str, password_hash: str) -> None:
    """Create a user, or replace an existing user's display name and password."""

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO app_users (username, display_name, password_hash)
            VALUES (?, ?, ?)
            ON CONFLICT(username) DO UPDATE SET
                display_name = excluded.display_name,
                password_hash = excluded.password_hash,
                is_active = 1
            """,
            (username.strip(), display_name.strip(), password_hash),
        )


def set_active(username: str, is_active: bool) -> bool:
    """Enable or disable a user. Returns False when no such user exists."""

    with connect() as conn:
        cursor = conn.execute(
            "UPDATE app_users SET is_active = ? WHERE username = ?",
            (1 if is_active else 0, username.strip()),
        )
        return cursor.rowcount > 0


def list_users() -> list[AppUser]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT user_id, username, display_name, is_active FROM app_users ORDER BY username"
        ).fetchall()
    return [
        AppUser(
            user_id=row["user_id"],
            username=row["username"],
            display_name=row["display_name"],
            is_active=bool(row["is_active"]),
        )
        for row in rows
    ]
