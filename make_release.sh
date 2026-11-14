#!/usr/bin/env bash
set -euo pipefail

final=submitted
if [[ ${1:-} == --final ]]; then
    final=${2:-}
    if (( $# >= 2 )); then shift 2; fi
fi
if (( $# < 1 || $# > 2 )) || [[ $final != public-ensemble && $final != submitted && $final != three-verifiers && $final != single-unet ]]; then
    echo "Usage: $0 [--final submitted|three-verifiers|public-ensemble|single-unet] WEIGHTS_DIR [RELEASE.zip]" >&2
    exit 2
fi

root=$(cd -- "$(dirname -- "$0")" && pwd)
"${PYTHON:-python3}" - "$root" "$1" "${2:-}" "$final" <<'PY'
"""Verify the release manifest and optionally write an inference-only ZIP."""

import hashlib
import os
from pathlib import Path
import re
import sys
import tempfile
import zipfile

root, weights = map(Path, sys.argv[1:3])
output = Path(sys.argv[3]) if sys.argv[3] else None
final = sys.argv[4]
manifest = root / ("release_weights_public_ensemble.sha256" if final == "public-ensemble" else "release_weights.sha256")


def copy_verified(source, digest, destination=None):
    """Hash the actual bytes being verified or written into the archive."""
    checksum = hashlib.sha256()
    with source.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
            if destination is not None:
                destination.write(block)
    actual = checksum.hexdigest()
    if actual != digest:
        raise ValueError(f"SHA-256 mismatch for {source.name}. Expected {digest}, got {actual}.")
    print(f"Verified {source.name}", flush=True)


def package(entries):
    """Publish the local archive only after all member hashes match."""
    if output.exists():
        raise FileExistsError(f"Archive already exists: {output}")
    sources = [Path(name) for name in (
        "README.md", "LICENSE", "WEIGHTS_LICENSE.md", manifest.name, "requirements.txt",
        "make_release.sh", "submit.py", "kernels/k07-final/final.py",
    )]
    if final == "public-ensemble":
        sources.extend(map(Path, ("pipeline.ipynb", "requirements-lock.txt", "tools/check_submission.py",
                                 "docs/REPRODUCE.md", "weights/public_ensemble/README.md", "weights/public/README.md")))
    else:
        sources.append(Path("kernels/release/submitted.ipynb"))
    sources.extend(path.relative_to(root) for path in sorted((root / "filseg").glob("*.py")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f"final-{final.lower()}-",
                                     suffix=".tmp", delete=False) as tmp:
        temporary = Path(tmp.name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for path in sources:
                archive.write(root / path, path.as_posix())
            for name, digest in entries:
                with archive.open(f"weights/{name}", "w") as stream:
                    copy_verified(weights / name, digest, stream)
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Wrote {output}")


try:
    entries = []
    checkpoint = r"[A-Za-z0-9_.-]+\.(?:pt|txt)"
    name_pattern = (rf"(?:(?:public_ensemble|public)/)?{checkpoint}|public/solar_unet_receipt\.json"
                    if final == "public-ensemble" else checkpoint)
    expected = 17 if final == "public-ensemble" else 6
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split()
        if not re.fullmatch(r"[a-f0-9]{64}", digest) or not re.fullmatch(name_pattern, name):
            raise ValueError(f"Invalid checksum entry for {name}.")
        entries.append((name, digest))
    if len(entries) != expected or len({name for name, _ in entries}) != expected:
        raise ValueError(f"Expected {expected} distinct artifact names for Final {final}.")
    if output is None:
        for name, digest in entries:
            copy_verified(weights / name, digest)
    else:
        package(entries)
except (OSError, ValueError) as exc:
    sys.exit(str(exc))
PY
