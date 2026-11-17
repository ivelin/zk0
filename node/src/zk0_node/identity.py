"""Node identity: an Ed25519 keypair owned by the node (spec §3 Stage 1).

The node key is what signs eval records, round records, join-gate
attestations, and bundles — a signature is the node's own act, made
with a key the node holds. Slice 1 anchored records with a SHA-256
content hash only; slice 2 adds the key layer on top (the content
hash stays as the inner layer — see records.py).

Implementation note: the Ed25519 below is the RFC 8032 reference
algorithm implemented in pure Python over hashlib's SHA-512. It is
deliberately dependency-free (the runtime manifest gains no new pin
for signing) and is verified against the RFC 8032 test vectors in
the package test suite. Signing a 32-byte record hash is fast;
verification is the same order. Key storage in v0.1 is a 0600 file
under the node's identity dir — stated plainly, not dressed up as
an HSM.
"""
import hashlib
import json
import os
from pathlib import Path

from .config import zk0_home

# ---------------------------------------------------------------- Ed25519
# RFC 8032 reference implementation (public-domain algorithm), operating
# on the twisted Edwards curve -x^2 + y^2 = 1 - (121665/121666) x^2 y^2
# over GF(2^255 - 19).

_Q = 2 ** 255 - 19
_L = 2 ** 252 + 27742317777372353535851937790883648493


def _inv(x):
    return pow(x, _Q - 2, _Q)


_D = (-121665 * _inv(121666)) % _Q
_I = pow(2, (_Q - 1) // 4, _Q)


def _xrecover(y):
    xx = (y * y - 1) * _inv(_D * y * y + 1)
    x = pow(xx, (_Q + 3) // 8, _Q)
    if (x * x - xx) % _Q != 0:
        x = (x * _I) % _Q
    if x % 2 != 0:
        x = _Q - x
    return x


_BY = (4 * _inv(5)) % _Q
_BX = _xrecover(_BY)
_B = (_BX % _Q, _BY % _Q)


def _edwards(P, Q):
    x1, y1 = P
    x2, y2 = Q
    denom = _inv(1 + _D * x1 * x2 * y1 * y2)
    x3 = (x1 * y2 + x2 * y1) * denom
    denom2 = _inv(1 - _D * x1 * x2 * y1 * y2)
    y3 = (y1 * y2 + x1 * x2) * denom2
    return x3 % _Q, y3 % _Q


def _scalarmult(P, e):
    Q = (0, 1)
    while e > 0:
        if e & 1:
            Q = _edwards(Q, P)
        P = _edwards(P, P)
        e >>= 1
    return Q


def _encodeint(y):
    bits = [(y >> i) & 1 for i in range(256)]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8))
                 for i in range(32))


def _encodepoint(P):
    x, y = P
    bits = [(y >> i) & 1 for i in range(255)] + [x & 1]
    return bytes(sum(bits[i * 8 + j] << j for j in range(8))
                 for i in range(32))


def _bit(h, i):
    return (h[i // 8] >> (i % 8)) & 1


def _publickey(seed32: bytes) -> bytes:
    h = hashlib.sha512(seed32).digest()
    a = 2 ** 254 + sum(2 ** i * _bit(h, i) for i in range(3, 254))
    A = _scalarmult(_B, a)
    return _encodepoint(A)


def _Hint(m: bytes) -> int:
    h = hashlib.sha512(m).digest()
    return sum(2 ** i * _bit(h, i) for i in range(512))


def _sign(seed32: bytes, msg: bytes) -> bytes:
    h = hashlib.sha512(seed32).digest()
    a = 2 ** 254 + sum(2 ** i * _bit(h, i) for i in range(3, 254))
    pub = _encodepoint(_scalarmult(_B, a))
    r = _Hint(h[32:64] + msg)
    R = _scalarmult(_B, r)
    S = (r + _Hint(_encodepoint(R) + pub + msg) * a) % _L
    return _encodepoint(R) + _encodeint(S)


def _decodeint(s: bytes) -> int:
    return sum(2 ** i * _bit(s, i) for i in range(256))


def _decodepoint(s: bytes):
    y = sum(2 ** i * _bit(s, i) for i in range(255))
    x = _xrecover(y)
    if x & 1 != _bit(s, 255):
        x = _Q - x
    return x, y


def _checkvalid(sig: bytes, msg: bytes, pub: bytes) -> bool:
    if len(sig) != 64:
        return False
    R = _decodepoint(sig[:32])
    A = _decodepoint(pub)
    S = _decodeint(sig[32:64])
    if S >= _L:
        return False
    h = _Hint(sig[:32] + pub + msg)
    return _scalarmult(_B, S) == _edwards(R, _scalarmult(A, h))


# ---------------------------------------------------------------- node key

def identity_dir() -> Path:
    return zk0_home() / "identity"


def key_path() -> Path:
    return identity_dir() / "node_key.json"


class NodeKey:
    """The node's Ed25519 keypair (seed kept in a 0600 file, v0.1)."""

    def __init__(self, seed: bytes):
        if len(seed) != 32:
            raise ValueError("ed25519 seed must be 32 bytes")
        self._seed = seed
        self.public_key = _publickey(seed)

    @property
    def public_key_hex(self) -> str:
        return self.public_key.hex()

    @property
    def fingerprint(self) -> str:
        """Short public identity of the key: sha256(pubkey)[:16]."""
        return hashlib.sha256(self.public_key).hexdigest()[:16]

    def sign(self, msg: bytes) -> bytes:
        return _sign(self._seed, msg)

    def sign_hex(self, msg: bytes) -> str:
        return self.sign(msg).hex()

    @staticmethod
    def verify(pubkey_hex: str, msg: bytes, sig_hex: str) -> bool:
        try:
            return _checkvalid(bytes.fromhex(sig_hex), msg,
                               bytes.fromhex(pubkey_hex))
        except Exception:
            return False


def load_or_create_key() -> NodeKey:
    """Load the node key, generating and persisting one on first use.

    Generation is a Stage 1 act: it happens at first signed record,
    first gate run, or first FL enrollment — whichever comes first —
    and the key then persists for the node's lifetime.
    """
    p = key_path()
    if p.exists():
        data = json.loads(p.read_text())
        return NodeKey(bytes.fromhex(data["seed_hex"]))
    seed = os.urandom(32)
    key = NodeKey(seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "alg": "ed25519",
        "seed_hex": seed.hex(),
        "public_key_hex": key.public_key_hex,
        "fingerprint": key.fingerprint,
        "storage_note": "v0.1 dev-grade storage: 0600 file in the node "
                        "identity dir; not an HSM or OS keychain",
    }, indent=1, sort_keys=True))
    os.chmod(p, 0o600)
    return key


def load_key() -> NodeKey | None:
    """The node key if it exists yet, else None (never creates)."""
    p = key_path()
    if not p.exists():
        return None
    data = json.loads(p.read_text())
    return NodeKey(bytes.fromhex(data["seed_hex"]))
