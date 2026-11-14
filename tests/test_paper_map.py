"""Every file linked from docs/PAPER_MAP.md exists in the repository."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_paper_map_links_exist():
    page = ROOT / "docs" / "PAPER_MAP.md"
    targets = re.findall(r"\]\(([^)#]+)(?:#[^)]*)?\)", page.read_text(encoding="utf-8"))
    assert targets
    missing = [t for t in targets if not (page.parent / t).resolve().exists()]
    assert not missing, missing
