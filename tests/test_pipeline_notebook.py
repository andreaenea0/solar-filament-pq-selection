"""Check that each notebook replay verifies its own release files."""

import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("finalist", ["public-ensemble", "three-verifiers", "submitted", "single-unet"])
def test_checkpoint_cell_uses_the_selected_manifest(tmp_path, finalist):
    notebook = json.loads((ROOT / "pipeline.ipynb").read_text(encoding="utf-8"))
    source = next(cell["source"] for cell in notebook["cells"] if cell["id"] == "checkpoints")
    manifest = "release_weights_public_ensemble.sha256" if finalist == "public-ensemble" else "release_weights.sha256"
    names = [row.split()[1] for row in (ROOT / manifest).read_text(encoding="ascii").splitlines()]
    weights = tmp_path / "weights"
    variant = weights / "public_ensemble" if finalist == "public-ensemble" else weights
    selector = ("lgb_pubown_nometa.txt" if finalist == "public-ensemble" else
                "own_ver3_a05_l025_nometa_lgb.txt")
    verifiers = [variant / f"verifier_resnet18_s{seed}.pt" for seed in range(3)]
    expected = [weights / f"effv2s_s{seed}_fold-1.pt" for seed in range(2)]
    expected += [variant / selector, *verifiers]
    if finalist == "public-ensemble":
        public = [f"unet_tu-tf_efficientnetv2_s_fold{fold}.pt" for fold in range(5)]
        public += [f"unet_resnet34_fold{fold}.pt" for fold in range(5)]
        expected += [weights / "public" / name for name in sorted(public)]
        expected.append(weights / "public" / "solar_unet_receipt.json")
    assert [weights / name for name in names] == expected

    payload = b"checkpoint fixture"
    for path in expected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / manifest).write_text("".join(f"{digest}  {name}\n" for name in names), encoding="ascii")
    receipt = weights / "public" / "solar_unet_receipt.json"
    if finalist == "public-ensemble":
        assert receipt.is_file()

    scope = {"ROOT": tmp_path, "RUN_TRAINING": False, "FINAL": finalist}
    exec("".join(source), scope)
    assert scope["selector"] == variant / selector
    one_verifier = finalist in {"submitted", "single-unet"}
    assert scope["verifier_files"] == (verifiers[:1] if one_verifier else verifiers)
    unets = [weights / f"effv2s_s{seed}_fold-1.pt" for seed in range(2)]
    assert scope["unet_files"] == (unets[:1] if finalist == "single-unet" else unets)
    assert scope["public_args"] == (["--public-receipt", receipt] if finalist == "public-ensemble" else [])
