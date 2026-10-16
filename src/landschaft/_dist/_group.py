# SPDX-License-Identifier: GPL-3.0-or-later
"""The group of computers that share work: its secret, kept in the user's configuration,
and the proofs that a coordinator and a helper both know it (or, while a helper joins,
the invitation code).

The short invitation code is never the long-term secret. It authenticates one pairing,
over TLS, in which the helper receives the group's random 256-bit secret. Every later
connection proves knowledge of that secret both ways, bound to the coordinator's TLS
certificate, so a machine in the middle can neither impersonate one side nor relay.
"""

from __future__ import annotations

import datetime
import functools
import hashlib
import hmac
import json
import os
import re
import secrets
import socket
import sys
from pathlib import Path

PROTOCOL = 1
TCP_PORT = 47811
UDP_PORT = 47810

_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"   # Crockford's base32: no I, L, O, U


def config_dir() -> Path:
    env = os.environ.get("LANDSCHAFT_CONFIG_DIR")
    if env:
        return Path(env)
    if sys.platform == "win32":
        return Path(os.environ.get("APPDATA") or Path.home()) / "landschaft"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "landschaft"


def _group_file() -> Path:
    return config_dir() / "group.json"


def load_secret() -> bytes | None:
    try:
        return bytes.fromhex(json.loads(_group_file().read_text())["secret"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_secret(secret: bytes) -> None:
    path = _group_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"secret": secret.hex()}, f)
    os.replace(tmp, path)


def ensure_secret() -> bytes:
    """The group's secret, created (a group of one) on first use."""
    secret = load_secret()
    if secret is None:
        secret = secrets.token_bytes(32)
        save_secret(secret)
    return secret


def forget_secret() -> bool:
    try:
        _group_file().unlink()
        return True
    except FileNotFoundError:
        return False


def group_id(secret: bytes) -> str:
    """A public name of the group, for discovery: it reveals nothing of the secret."""
    return hmac.new(secret, b"landschaft-group-id", hashlib.sha256).hexdigest()[:16]


def new_code() -> str:
    chars = "".join(secrets.choice(_ALPHABET) for _ in range(8))   # 40 bits
    return f"{chars[:4]}-{chars[4:]}"


def normalize_code(code: str) -> str:
    text = code.upper().replace("-", "").replace(" ", "").translate(str.maketrans("OIL", "011"))
    if len(text) != 8 or any(c not in _ALPHABET for c in text):
        raise ValueError(f"not an invitation code: {code!r} (it looks like K7QM-2XDP)")
    return text


def code_key(code: str, salt: bytes) -> bytes:
    """The pairing key: scrypt makes guessing the code offline from a captured proof
    take months instead of minutes, and the invitation expires long before."""
    return hashlib.scrypt(normalize_code(code).encode(), salt=salt, n=2 ** 15, r=8, p=1,
                          maxmem=2 ** 26, dklen=32)


def proof(key: bytes, role: str, *parts: str) -> str:
    message = "|".join(("landschaft-dist", str(PROTOCOL), role) + parts).encode()
    return hmac.new(key, message, hashlib.sha256).hexdigest()


def same(a: str, b: str) -> bool:
    return hmac.compare_digest(str(a), str(b))


def _checkout() -> Path | None:
    """The repository root when landschaft runs from a source checkout (an editable install)."""
    root = Path(__file__).resolve().parents[3]
    try:
        text = (root / "pyproject.toml").read_text()
    except OSError:
        return None
    return root if re.search(r'^name\s*=\s*"landschaft"', text, re.M) else None


@functools.lru_cache(maxsize=None)
def version() -> str:
    """landschaft's version. In a source checkout the installed metadata can be stale,
    so the checkout's pyproject.toml is read instead."""
    root = _checkout()
    if root is not None:
        found = re.search(r'^version\s*=\s*"([^"]+)"', (root / "pyproject.toml").read_text(), re.M)
        if found:
            return found.group(1)
    try:
        from importlib.metadata import version as _version

        return _version("landschaft")
    except Exception:  # noqa: BLE001
        return "unknown"


@functools.lru_cache(maxsize=None)
def code_fingerprint() -> str:
    """A hash of landschaft's Python code: two computers with the same version number but
    other code (a development install on one of them) could give other results. The same
    for a source checkout and an installed wheel of the same code (the compiled module
    differs between systems, so it is not part of it)."""
    package = Path(__file__).resolve().parents[1]
    h = hashlib.sha256()
    for path in sorted((p for p in package.rglob("*.py") if "__pycache__" not in p.parts),
                       key=lambda p: p.relative_to(package).as_posix()):
        h.update(path.relative_to(package).as_posix().encode() + bytes(1))
        h.update(path.read_bytes().replace(bytes([13, 10]), bytes([10])) + bytes(1))
    return h.hexdigest()[:16]


def machine_name() -> str:
    return socket.gethostname()


def certificate(directory: Path) -> tuple[str, str, str]:
    """A throwaway self-signed certificate: (cert file, key file, SHA-256 fingerprint).
    Its trust comes from the proofs, which are bound to the fingerprint, not from a CA."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "landschaft")])
    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number()).not_valid_before(now - datetime.timedelta(days=1))
            .not_valid_after(now + datetime.timedelta(days=30)).sign(key, hashes.SHA256()))
    directory.mkdir(parents=True, exist_ok=True)
    cert_path, key_path = directory / "cert.pem", directory / "key.pem"
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                  serialization.NoEncryption()))
    fingerprint = hashlib.sha256(cert.public_bytes(serialization.Encoding.DER)).hexdigest()
    return str(cert_path), str(key_path), fingerprint
