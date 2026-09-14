"""Role-helper contract: validate_inputs (validation stage) and evaluation_report (evaluation stage)."""

from __future__ import annotations

import pytest
from PIL import Image

from pix2struct_ai2d_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_OPTIONS,
    MAX_QUESTION_CHARS,
    MIN_IMAGE_SIDE,
    MIN_OPTIONS,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    format_prompt,
    match_option,
    normalize_answer,
    validate_inputs,
)

OPTIONS = ["flower", "leaf", "stem", "root"]
QUESTIONS = [
    {"question": "What does the label 1 represent?", "options": OPTIONS},
    {"question": "What provides light to the plant?", "options": ["soil", "sun"]},
]


def _image(width: int = 640, height: int = 640) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(answer: str, options: list[str] = OPTIONS, question: str = "Q?", truncated: bool = False) -> dict:
    return {
        "answer": answer,
        "choice_index": match_option(answer, options),
        "question": question,
        "options": list(options),
        "truncated": truncated,
    }


def test_format_prompt_and_match_option():
    assert format_prompt("What is 1?", ["lava", "ash cloud"]) == "What is 1? (1) lava (2) ash cloud"
    assert normalize_answer("  Ash-Cloud. ") == "ash cloud"
    assert match_option(" Ash cloud.", ["lava", "ash cloud"]) == 1
    assert match_option("ash", ["lava", "ash cloud"]) is None  # no fuzzy or prefix match
    assert match_option("root root (2) root", OPTIONS) is None


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), QUESTIONS, names=["plant.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["question_chars"] == [1, MAX_QUESTION_CHARS]
    assert manifest["schema"]["options"] == [MIN_OPTIONS, MAX_OPTIONS]
    assert manifest["inputs"] == [{"id": "plant.png", "mode": "RGB", "size": [640, 640]}]
    assert manifest["questions"][1] == {
        "question": "What provides light to the plant?",
        "options": ["soil", "sun"],
        "prompt": "What provides light to the plant? (1) soil (2) sun",
    }
    assert manifest["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_request() -> None:
    manifest = validate_inputs(
        _image(), [{"question": "  Which   part? ", "options": [" a ", "b"]}], max_new_tokens=8
    )
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["questions"][0]["question"] == "Which part?"
    assert manifest["questions"][0]["options"] == ["a", "b"]
    assert manifest["generation"]["max_new_tokens"] == 8


def test_validate_inputs_rejects_like_answer() -> None:
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs(_image(), QUESTIONS[0])
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs(_image(), [])
    with pytest.raises(TypeError, match="mapping with"):
        validate_inputs(_image(), [{"question": "Q?"}])
    with pytest.raises(ValueError, match="MIN_OPTIONS"):
        validate_inputs(_image(), [{"question": "Q?", "options": ["one"]}])
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(_image(), QUESTIONS, max_new_tokens=0)
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8), QUESTIONS)
    with pytest.raises(ValueError, match="exactly one entry"):
        validate_inputs(_image(), QUESTIONS, names=["a", "b"])


def test_evaluation_report_not_measurable_without_labels() -> None:
    report = evaluation_report([_result("root")], sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == [] and report["baselines"] == []
    assert report["n_questions"] == 1 and report["truncated"] == [False] and report["unmatched"] == [False]
    assert "accuracy" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "no score" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_labels() -> None:
    results = [
        _result("root"),  # gold 0 -> wrong
        _result("sun", ["soil", "sun"], truncated=True),  # gold 1 -> correct
        _result("root root (2) root"),  # unmatched -> wrong
        _result("root"),  # gold 3 -> correct
    ]
    report = evaluation_report(results, [0, 1, 2, 3])
    assert report["verdict"] == "sample-sanity"
    assert report["truncated"] == [False, True, False, False]
    assert report["unmatched"] == [False, False, True, False]
    by_id = {metric["id"]: metric for metric in report["metrics"]}
    assert by_id["accuracy"]["value"] == 0.5
    assert by_id["unmatched_rate"]["value"] == 0.25
    assert report["baselines"][0]["id"] == "chance"
    assert report["baselines"][0]["value"] == pytest.approx((0.25 + 0.5 + 0.25 + 0.25) / 4)
    assert [entry["correct"] for entry in report["per_question"]] == [False, True, False, True]
    assert report["per_question"][0]["correct_option"] == "flower"


def test_evaluation_report_rejects_mismatched_or_invalid_labels() -> None:
    with pytest.raises(ValueError, match="correct has"):
        evaluation_report([_result("root")], [0, 1])
    with pytest.raises(ValueError, match="valid zero-based"):
        evaluation_report([_result("root")], [4])
    with pytest.raises(ValueError, match="valid zero-based"):
        evaluation_report([_result("root")], [True])
    with pytest.raises(ValueError, match="results"):
        evaluation_report([], None)
