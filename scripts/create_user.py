"""Create, update or list app login users.

There is no self-service signup on purpose - accounts are created here by
someone with access to the server:

    python scripts/create_user.py tech1 --display-name "Tech One"   # random password
    python scripts/create_user.py tech1 --password 'chosen-password'
    python scripts/create_user.py --list
    python scripts/create_user.py tech1 --disable

Generated passwords are printed once and stored only as PBKDF2 hashes.
"""

import argparse
import secrets
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.auth.passwords import hash_password  # noqa: E402
from app.auth.users import list_users, set_active, upsert_user  # noqa: E402


def generate_password(length: int = 14) -> str:
    # Ambiguous characters removed - these get typed on tablet keyboards, where
    # 0/O and 1/l/I cause real support calls.
    alphabet = "".join(c for c in (string.ascii_letters + string.digits) if c not in "0O1lI")
    return "".join(secrets.choice(alphabet) for _ in range(length))


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage app login users.")
    parser.add_argument("username", nargs="?", help="Username to create or update.")
    parser.add_argument("--display-name", help="Name shown in the UI (defaults to the username).")
    parser.add_argument("--password", help="Explicit password (default: generate one).")
    parser.add_argument("--disable", action="store_true", help="Disable the user instead.")
    parser.add_argument("--list", action="store_true", help="List users and exit.")
    args = parser.parse_args()

    if args.list:
        users = list_users()
        if not users:
            print("(no users)")
        for user in users:
            state = "active" if user.is_active else "disabled"
            print(f"  {user.user_id:>3}  {user.username:<22} {user.display_name:<24} {state}")
        return

    if not args.username:
        parser.error("a username is required unless --list is given")

    if args.disable:
        if not set_active(args.username, False):
            raise SystemExit(f"{args.username}: not found")
        print(f"{args.username}: disabled")
        return

    password = args.password or generate_password()
    upsert_user(args.username, args.display_name or args.username, hash_password(password))
    print(f"{args.username}: {password}")


if __name__ == "__main__":
    main()
