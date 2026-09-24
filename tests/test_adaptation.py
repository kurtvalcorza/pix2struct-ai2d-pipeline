"""Offline checks of the multiple-choice dataset contract, the metrics and baselines, the pinned-shard
reader (refusal while unpinned, digest checks when pinned) and the adaptation surface — no model weights,
no network."""

from __future__ import annotations

import hashlib
import io
import json

import pytest
from PIL import Image

from pix2struct_ai2d_pipeline import (
    MAX_RECORDS,
    MIN_RECORDS,
    Pix2StructAI2DPipeline,
    build_sample_dataset,
    check_split_disjoint,
    corpus_pinned,
    dataset_digest,
    fetch_corpus,
    load_byod_dataset,
    longest_option_baseline,
    mcq_metrics,
    option_category,
    position_prior_baseline,
    read_corpus,
    split_dataset,
    validate_dataset,
    write_dataset_jsonl,
)
from pix2struct_ai2d_pipeline import samples as samples_module

COLOURS = ["red", "green", "blue", "yellow", "purple", "orange"]


def _png(colour: str, size=(64, 48)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def records(tmp_path):
    out = []
    for i in range(12):
        path = tmp_path / f"d{i // 2}.png"
        if not path.is_file():
            path.write_bytes(_png(COLOURS[(i // 2) % len(COLOURS)]))
        out.append(
            {
                "id": f"q{i:02d}",
                "image_id": f"d{i // 2}",
                "image": str(path),
                "question": f"Which colour is diagram {i // 2}?",
                "options": ["red", "green", "blue", "yellow"],
                "answer": i % 4,
                "category": "text-option",
            }
        )
    return out


# ---- metrics and baselines ------------------------------------------------------------------------------


def test_mcq_metrics_counts_unmatched_as_wrong_and_reports_chance():
    metrics = mcq_metrics([0, None, 2, 1], [0, 1, 2, 3], [4, 4, 3, 4], categories=["a", "a", "b", "b"])
    assert metrics["n"] == 4 and metrics["accuracy"] == 0.5 and metrics["unmatched_rate"] == 0.25
    assert metrics["chance"] == pytest.approx((0.25 + 0.25 + 1 / 3 + 0.25) / 4)
    assert metrics["by_category"] == {"a": {"n": 2, "accuracy": 0.5}, "b": {"n": 2, "accuracy": 0.5}}
    assert set(metrics["definitions"]) >= {"accuracy", "unmatched_rate", "chance"}


@pytest.mark.parametrize(
    ("predicted", "correct", "n_options"),
    [([], [], []), ([0], [0, 1], [2, 2]), ([0], [4], [4]), ([0], [True], [2])],
)
def test_mcq_metrics_refuses_misaligned_or_invalid_labels(predicted, correct, n_options):
    with pytest.raises(ValueError):
        mcq_metrics(predicted, correct, n_options)


def test_position_prior_baseline_uses_the_training_mode_and_clamps(records):
    train = [{**r, "answer": 3} for r in records[:6]]
    test = [{**records[6], "options": ["a", "b"], "answer": 1}, {**records[7], "answer": 3}]
    result = position_prior_baseline(train, test)
    assert result["position"] == 3 and result["baseline"] == "position-prior"
    assert result["accuracy"] == 1.0  # clamped to option 2 of 2 on the first, option 4 on the second


def test_longest_option_baseline_picks_the_longest_first_on_ties(records):
    test = [
        {**records[0], "options": ["ab", "abcd", "abc"], "answer": 1},
        {**records[1], "options": ["xyz", "uvw"], "answer": 1},
    ]
    result = longest_option_baseline(test)
    assert result["accuracy"] == 0.5 and result["baseline"] == "longest-option"


def test_option_category_separates_diagram_labels_from_text():
    assert option_category(["A", "b", "12"]) == "letter-label"
    assert option_category(["A", "leaf"]) == "text-option"


# ---- dataset contract -----------------------------------------------------------------------------------


def test_validate_dataset_reports_structure_and_digest(records):
    manifest = validate_dataset(records)
    assert manifest["n_records"] == 12 and manifest["unique_images"] == 6
    assert manifest["options_per_question"] == {"min": 4, "max": 4}
    assert manifest["answer_positions"] == {0: 3, 1: 3, 2: 3, 3: 3}
    assert manifest["digest"] == dataset_digest(manifest["records"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: {**r, "id": "bad id!"}, "id must match"),
        (lambda r: {**r, "image": "/does/not/exist.png"}, "image file not found"),
        (lambda r: {**r, "question": "   "}, "question must be a non-empty"),
        (lambda r: {**r, "question": "x" * 257}, "MAX_QUESTION_CHARS"),
        (lambda r: {**r, "options": ["only"]}, "options must have"),
        (lambda r: {**r, "options": ["Leaf", "leaf."]}, "distinct"),
        (lambda r: {**r, "options": ["a", ""]}, "non-empty string"),
        (lambda r: {**r, "answer": 4}, "zero-based index"),
        (lambda r: {**r, "answer": "1"}, "zero-based index"),
        (lambda r: {k: v for k, v in r.items() if k != "answer"}, "missing 'answer'"),
    ],
)
def test_validate_dataset_refuses_contract_violations(records, mutate, message):
    broken = [mutate(records[0]), *records[1:]]
    with pytest.raises(ValueError, match=message):
        validate_dataset(broken)


def test_validate_dataset_refuses_duplicates_and_size_bounds(records):
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], *records[:8]])
    with pytest.raises(ValueError, match=f"{MIN_RECORDS}..{MAX_RECORDS}"):
        validate_dataset(records[:3])


def test_split_dataset_keeps_each_diagram_in_one_split(records):
    splits = split_dataset(records * 1, val_fraction=0.2, test_fraction=0.2, seed=1)
    counts = check_split_disjoint(splits)
    assert sum(counts.values()) == 12
    for name, part in splits.items():
        others = {r["image_id"] for other, rest in splits.items() if other != name for r in rest}
        assert not {r["image_id"] for r in part} & others


def test_check_split_disjoint_refuses_a_shared_diagram(records):
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": records[:1], "test": records[1:2]})


def test_byod_jsonl_round_trip(records, tmp_path):
    path = write_dataset_jsonl(records, tmp_path / "out" / "records.jsonl")
    again = load_byod_dataset(path)
    assert [r["id"] for r in again] == [r["id"] for r in records]
    assert validate_dataset(again)["digest"] == validate_dataset(records)["digest"]


# ---- pinned shard ---------------------------------------------------------------------------------------


def _shard(tmp_path, n_diagrams=40, per_diagram=3):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for d in range(n_diagrams):
        data = _png((d * 5 % 255, 40, 200 - d), size=(32 + d, 32))
        for q in range(per_diagram):
            options = ["A", "B", "C", "D"] if q == 0 else ["leaf", "stem", "root", "flower"]
            image = {"bytes": data, "path": None}
            rows.append({"question": f"q{d}-{q}", "options": options, "answer": str(q % 4), "image": image})
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


def test_fetch_corpus_refuses_while_no_pin_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", None)
    assert not corpus_pinned()
    with pytest.raises(RuntimeError, match="no SHA-256 pin is recorded"):
        fetch_corpus(cache_dir=tmp_path, downloader=lambda cache: pytest.fail("must not download"))


def test_fetch_corpus_checks_size_and_digest(tmp_path, monkeypatch):
    shard = _shard(tmp_path)
    data = shard.read_bytes()
    monkeypatch.setitem(samples_module.CORPUS_FILE, "bytes", len(data))
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", hashlib.sha256(data).hexdigest())
    assert corpus_pinned()
    assert fetch_corpus(cache_dir=tmp_path / "cache", downloader=lambda cache: shard) == shard
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", "0" * 64)
    with pytest.raises(ValueError, match="refusing to read it"):
        fetch_corpus(cache_dir=tmp_path / "cache", downloader=lambda cache: shard)


def test_read_corpus_and_build_sample_split_by_diagram(tmp_path, monkeypatch):
    shard = _shard(tmp_path)
    monkeypatch.setitem(samples_module.CORPUS_FILE, "rows", None)
    monkeypatch.setitem(samples_module.CORPUS_FILE, "images", None)
    rows = read_corpus(shard)
    assert len(rows) == 120 and rows[1]["answer"] == 1 and rows[0]["options"] == ["A", "B", "C", "D"]
    sizes = {"train": 60, "validation": 15, "test": 30}
    splits = build_sample_dataset(rows, seed=3, sizes=sizes, image_dir=tmp_path / "images")
    counts = check_split_disjoint(splits)
    assert all(counts[name] >= sizes[name] for name in sizes)
    assert all(counts[name] % 3 == 0 for name in sizes)  # whole diagrams only
    again = build_sample_dataset(rows, seed=3, sizes=sizes, image_dir=tmp_path / "images")
    assert [r["id"] for r in again["test"]] == [r["id"] for r in splits["test"]]
    assert {r["category"] for r in splits["train"]} == {"letter-label", "text-option"}
    for part in splits.values():
        validate_dataset(part)
    monkeypatch.setitem(samples_module.CORPUS_FILE, "images", 39)
    with pytest.raises(ValueError, match="pinned 39"):
        build_sample_dataset(rows, sizes=sizes, image_dir=tmp_path / "images")
    monkeypatch.setitem(samples_module.CORPUS_FILE, "images", None)
    with pytest.raises(ValueError, match="do not fill"):
        too_many = {"train": 200, "validation": 1, "test": 1}
        build_sample_dataset(rows, sizes=too_many, image_dir=tmp_path / "i2")


# ---- adaptation surface without a model -------------------------------------------------------------------


def _runner(image, prompt, max_new_tokens):
    # Answers the first option of the rendered prompt: "<question> (1) <a> (2) <b> ..."
    first = prompt.split("(1) ", 1)[1].split(" (2)", 1)[0]
    return {"answer": first, "new_tokens": 2}


def test_evaluate_with_an_injected_runner_scores_accuracy(records):
    pipe = Pix2StructAI2DPipeline(_runner)
    metrics = pipe.evaluate(records)
    assert metrics["n"] == 12 and metrics["accuracy"] == pytest.approx(3 / 12)
    assert metrics["unmatched_rate"] == 0.0 and metrics["verdict"] == "measured-small-sample"
    assert metrics["adapted"] is False
    predictions = pipe.predict(records[:2])
    assert [p["choice_index"] for p in predictions] == [0, 0] and predictions[0]["id"] == "q00"


def test_adaptation_needs_a_loaded_model(records, tmp_path):
    pipe = Pix2StructAI2DPipeline(_runner)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(records)
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    with pytest.raises(ValueError, match="1..12"):
        pipe._trainable_names(13)


def test_load_artifact_refuses_a_foreign_manifest(tmp_path):
    pipe = Pix2StructAI2DPipeline(_runner)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
