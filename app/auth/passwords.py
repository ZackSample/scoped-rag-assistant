"""Password hashing for the app's own login screen.

Deliberately uses PBKDF2 from the standard library rather than bcrypt/argon2:
no extra dependency, nothing to compile in the container image, and it is a
genuinely acceptable KDF when the iteration count is high enough. Stored hashes
are one-way, so a leaked database row does not hand someone a working login.
"""

import hashlib
import hmac
import os

# Format: pbkdf2_sha256$<iterations>$<salt_hex>$<hash_hex>
_ALGORITHM = "pbkdf2_sha256"
_ITERATIONS = 240_000
_SALT_BYTES = 16


def hash_password(password: str, *, iterations: int = _ITERATIONS) -> str:
    """Hash a plaintext password into a self-describing storable string."""

    salt = os.urandom(_SALT_BYTES)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_ALGORITHM}${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Check a plaintext password against a stored hash.

    Returns False rather than raising on malformed input so a corrupted row
    can't turn into a 500 on the login endpoint.
    """

    if not stored:
        return False
    try:
        algorithm, iterations_raw, salt_hex, digest_hex = stored.split("$")
        if algorithm != _ALGORITHM:
            return False
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations_raw)
        )
    except (ValueError, TypeError):
        return False
    # compare_digest keeps the check constant-time so response timing doesn't
    # leak how much of the hash matched.
    return hmac.compare_digest(expected, actual)
