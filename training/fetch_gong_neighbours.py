"""Fetch unlabeled GONG H-alpha frames around annotated training observations."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from http.client import HTTPException
from pathlib import Path
import tempfile
import urllib.request


def job(stem, offset, output):
    """Fetch one offset and publish it only after the full response is saved."""
    time = datetime.strptime(stem[:14], "%Y%m%d%H%M%S") + timedelta(minutes=offset)
    name = time.strftime("%Y%m%d%H%M%S") + stem[14:]
    directory = output / stem
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{offset:+d}_{name}.jpg"
    if destination.exists():
        return "skip"

    url = f"https://gong2.nso.edu/HA/hag/{time:%Y%m}/{time:%Y%m%d}/{name}.jpg"
    temporary = None
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
        temporary.replace(destination)
    except (OSError, HTTPException):
        return "miss"
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return "ok"


def main():
    """Download the requested offsets and print the existing status counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offsets", default="-6,-4,-2,2,4,6")
    args = parser.parse_args()
    offsets = [int(value) for value in args.offsets.split(",")]
    stems = sorted(path.stem for path in args.images.glob("*.jpeg"))
    jobs = [(stem, offset) for stem in stems for offset in offsets]
    with ThreadPoolExecutor(16) as executor:
        counts = Counter(executor.map(lambda pair: job(*pair, args.out), jobs))
    print(counts)


if __name__ == "__main__":
    main()
