import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pix2struct_ai2d_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_WEIGHTS_DIR,
    MAX_IMAGE_SIDE,
    MAX_NEW_TOKENS,
    MAX_OPTION_CHARS,
    MAX_OPTIONS,
    MAX_PATCHES,
    MAX_QUESTION_CHARS,
    MIN_IMAGE_SIDE,
    MIN_OPTIONS,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    Pix2StructAI2DPipeline,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]
OPTIONS = ["flower", "leaf", "stem", "root"]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "google/pix2struct-ai2d-base"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert 1 <= DEFAULT_MAX_NEW_TOKENS <= MAX_NEW_TOKENS == 64
    assert MAX_PATCHES == 2048 and MAX_QUESTION_CHARS == 256 and MAX_OPTION_CHARS == 64
    assert 2 <= MIN_OPTIONS < MAX_OPTIONS == 6
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION
        paths = [entry["path"] for entry in data["files"]]
        assert "model.safetensors" in paths and "pytorch_model.bin" not in paths


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "pix2struct"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "pix2struct"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


DOCTAGS = (
    "<doctag><section_header_level_1><loc_32><loc_25><loc_231><loc_39>Report</section_header_level_1>\n"
    "<text><loc_32><loc_40><loc_319><loc_93>quarterly revenue by region</text>\n"
    "<otsl><loc_32><loc_151><loc_459><loc_303><ched>Region<ched>Q1<nl><fcel>North 1<fcel>21,132<nl></otsl>\n"
    "</doctag>"
)


def _fake_pipeline(calls: list | None = None) -> Pix2StructAI2DPipeline:
    def runner(image, prompt, max_new_tokens):
        if calls is not None:
            calls.append((image.mode, prompt, max_new_tokens))
        answer = " Root " if "label 4" in prompt else "root root (2) root"
        return {"answer": answer, "new_tokens": 2}

    return Pix2StructAI2DPipeline(runner, "cpu", "float32", "injected")


def test_answer_output_fields_prompt_and_option_match():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.answer(
        Image.new("L", (400, 300)),
        "  What does the   label 4 represent? ",
        ["stem", " flower ", "root", "leaf"],
    )
    assert result["answer"] == "Root"  # stripped, case kept
    assert result["choice_index"] == 2  # normalised exact match against the checked options
    assert result["question"] == "What does the label 4 represent?"  # whitespace collapsed
    assert result["options"] == ["stem", "flower", "root", "leaf"]
    assert result["prompt"] == "What does the label 4 represent? (1) stem (2) flower (3) root (4) leaf"
    assert result["image_size"] == [400, 300]
    assert result["new_tokens"] == 2 and result["truncated"] is False
    assert result["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert (result["device"], result["dtype"], result["source"]) == ("cpu", "float32", "injected")
    assert calls == [("RGB", result["prompt"], DEFAULT_MAX_NEW_TOKENS)]


def test_answer_unmatched_text_yields_null_index_and_truncation_flag():
    pipe = _fake_pipeline()
    result = pipe.answer(
        Image.new("RGB", (64, 64)), "What does the label 1 represent?", OPTIONS, max_new_tokens=2
    )
    assert result["answer"] == "root root (2) root" and result["choice_index"] is None
    assert result["truncated"] is True


def test_answer_rejects_bad_inputs():
    pipe = _fake_pipeline()
    image = Image.new("RGB", (64, 64))
    with pytest.raises(TypeError):
        pipe.answer(np.zeros((30, 40, 3), dtype=np.uint8), "Q?", OPTIONS)
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.answer(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)), "Q?", OPTIONS)
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.answer(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)), "Q?", OPTIONS)
    with pytest.raises(TypeError, match="question must be a str"):
        pipe.answer(image, None, OPTIONS)
    with pytest.raises(ValueError, match="non-whitespace"):
        pipe.answer(image, "   ", OPTIONS)
    with pytest.raises(ValueError, match="MAX_QUESTION_CHARS"):
        pipe.answer(image, "x" * (MAX_QUESTION_CHARS + 1), OPTIONS)
    with pytest.raises(TypeError, match="options must be a sequence"):
        pipe.answer(image, "Q?", "flower leaf")
    with pytest.raises(ValueError, match="MIN_OPTIONS"):
        pipe.answer(image, "Q?", ["flower"])
    with pytest.raises(ValueError, match="MAX_OPTIONS"):
        pipe.answer(image, "Q?", [f"o{i}" for i in range(MAX_OPTIONS + 1)])
    with pytest.raises(TypeError, match="each option must be a str"):
        pipe.answer(image, "Q?", ["flower", 3])
    with pytest.raises(ValueError, match="each option must contain"):
        pipe.answer(image, "Q?", ["flower", "  "])
    with pytest.raises(ValueError, match="MAX_OPTION_CHARS"):
        pipe.answer(image, "Q?", ["flower", "x" * (MAX_OPTION_CHARS + 1)])
    with pytest.raises(ValueError, match="distinct"):
        pipe.answer(image, "Q?", ["Flower", "flower."])
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        pipe.answer(image, "Q?", OPTIONS, max_new_tokens=MAX_NEW_TOKENS + 1)
    with pytest.raises(TypeError, match="max_new_tokens"):
        pipe.answer(image, "Q?", OPTIONS, max_new_tokens=True)


def test_answer_rejects_malformed_runner_output():
    pipe = Pix2StructAI2DPipeline(lambda *args: {"tokens": 1}, "cpu")
    with pytest.raises(RuntimeError, match="answer"):
        pipe.answer(Image.new("RGB", (64, 64)), "Q?", OPTIONS)
