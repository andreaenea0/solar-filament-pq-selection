"""Keep incomplete training downloads retryable."""

from http.client import IncompleteRead
from io import BytesIO
from pathlib import Path

from training import fetch_gong_neighbours as neighbours


def test_failed_download_can_be_retried(tmp_path, monkeypatch):
    stem = "20110120105534Ch"
    destination = tmp_path / stem / "+2_20110120105734Ch.jpg"

    class Truncated(BytesIO):
        def read(self):
            raise IncompleteRead(b"partial")

    monkeypatch.setattr(neighbours.urllib.request, "urlopen", lambda *args, **kwargs: Truncated())
    assert neighbours.job(stem, 2, tmp_path) == "miss"
    assert list(destination.parent.iterdir()) == []

    monkeypatch.setattr(neighbours.urllib.request, "urlopen", lambda *args, **kwargs: BytesIO(b"image bytes"))
    with monkeypatch.context() as failed_publish:
        def fail(self, target):
            raise OSError("failed rename")

        failed_publish.setattr(Path, "replace", fail)
        assert neighbours.job(stem, 2, tmp_path) == "miss"
        assert list(destination.parent.iterdir()) == []

    assert neighbours.job(stem, 2, tmp_path) == "ok"
    assert destination.read_bytes() == b"image bytes"
    assert neighbours.job(stem, 2, tmp_path) == "skip"
