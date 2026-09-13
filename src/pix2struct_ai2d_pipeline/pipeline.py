"""Diagram multiple-choice question answering with the pinned ``google/pix2struct-ai2d-base`` checkpoint.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the Pix2Struct architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed. The question and its numbered
options are rendered as a text header on top of the diagram (the Pix2Struct VQA input convention) with
Pillow's bundled font, so no font is fetched from the Hub at inference time; the generated answer text
is matched to the options by normalised exact match.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFont

MODEL_ID = "google/pix2struct-ai2d-base"
MODEL_REVISION = "0d6b2606efe05c77c1d0670647740802a9e68eef"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "pix2struct-ai2d-base"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Generation ceilings. AI2D answers are one option's text (the checkpoint's text_config max_length is
# 20); the default leaves room for a long option, the ceiling bounds runaway generation.
MAX_NEW_TOKENS = 64
DEFAULT_MAX_NEW_TOKENS = 16
DECODING = "greedy"
# Prompt ceilings. The question and the numbered options are rendered as header lines (wrapped at 80
# characters by the processor) above the diagram; a long prompt shrinks the diagram's share of the
# patch budget. AI2D questions carry four options; the ceiling allows a few more.
MAX_QUESTION_CHARS = 256
MAX_OPTION_CHARS = 64
MIN_OPTIONS = 2
MAX_OPTIONS = 6
# Input ceilings. The processor extracts at most MAX_PATCHES 16x16 patches (preprocessor_config.json)
# after scaling the image to fill that budget, so pixel count only guards memory during resizing.
MAX_PATCHES = 2048
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
_PUNCT_RE = re.compile(r"[^\w\s]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def header_font_bytes() -> bytes:
    """Pillow's bundled Aileron Regular (CC0) as TrueType bytes: the header font for the rendered question.

    The upstream image processor otherwise fetches ``ybelkada/fonts/Arial.TTF`` from the Hub at
    inference time — an unpinned, unlisted download of a proprietary font. The bundled subset covers
    the printable ASCII range, which is what a question is expected to use.
    """
    font = ImageFont.load_default(size=36)
    data = getattr(font, "font_bytes", None)
    if not data:
        raise RuntimeError("Pillow's bundled TrueType font is unavailable (FreeType support missing)")
    return bytes(data)


def normalize_answer(text: str) -> str:
    """Normalisation for option matching: lower-case, punctuation removed, whitespace collapsed."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def format_prompt(question: str, options: Sequence[str]) -> str:
    """The pinned README's AI2D prompt convention: ``<question> (1) <a> (2) <b> ...``."""
    return " ".join([question] + [f"({index}) {option}" for index, option in enumerate(options, start=1)])


def match_option(answer: str, options: Sequence[str]) -> int | None:
    """Zero-based index of the option the normalised answer equals, else ``None`` (no fuzzy match)."""
    pred = normalize_answer(answer)
    for index, option in enumerate(options):
        if pred == normalize_answer(option):
            return index
    return None


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one diagram image as PIL.Image.Image (any mode, converted to RGB) plus one question string and "
        "MIN_OPTIONS..MAX_OPTIONS answer options"
    ),
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "question_chars": [1, MAX_QUESTION_CHARS],
    "options": [MIN_OPTIONS, MAX_OPTIONS],
    "option_chars": [1, MAX_OPTION_CHARS],
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "decoding": f"{DECODING} (do_sample=False), deterministic on a fixed device and dtype",
    "preprocessing": (
        "the question and the numbered options are rendered as a black-on-white header (Pillow's bundled "
        "font, wrapped at 80 characters) above the diagram; the composite is scaled to fill at most "
        "MAX_PATCHES 16x16 patches (aspect ratio preserved), normalised per image and flattened into patch "
        "tokens with row/column positions; the decoder generates the answer text, which is matched to the "
        "options by normalised exact match"
    ),
    "output": "one answer string (the model's decoded text) plus the matched option index or null; no score",
}


def _check_inputs(
    image: Any, question: Any, options: Any, max_new_tokens: Any
) -> tuple[Image.Image, str, list[str], int]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``answer`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    if not isinstance(question, str):
        raise TypeError("question must be a str")
    checked_question = " ".join(question.split())
    if not checked_question:
        raise ValueError("question must contain at least one non-whitespace character")
    if len(checked_question) > MAX_QUESTION_CHARS:
        raise ValueError(
            f"question has {len(checked_question)} chars > MAX_QUESTION_CHARS {MAX_QUESTION_CHARS}"
        )
    if isinstance(options, str) or not isinstance(options, Sequence):
        raise TypeError("options must be a sequence of str")
    if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"options must have MIN_OPTIONS={MIN_OPTIONS}..MAX_OPTIONS={MAX_OPTIONS} entries")
    checked_options = []
    for option in options:
        if not isinstance(option, str):
            raise TypeError("each option must be a str")
        checked = " ".join(option.split())
        if not checked:
            raise ValueError("each option must contain at least one non-whitespace character")
        if len(checked) > MAX_OPTION_CHARS:
            raise ValueError(f"option has {len(checked)} chars > MAX_OPTION_CHARS {MAX_OPTION_CHARS}")
        checked_options.append(checked)
    if len({normalize_answer(option) for option in checked_options}) != len(checked_options):
        raise ValueError("options must be distinct after normalisation")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    return rgb, checked_question, checked_options, max_new_tokens


def validate_inputs(
    image: Image.Image,
    questions: Sequence[Mapping[str, Any]],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    ``questions`` holds mappings with ``question`` and ``options``; each is checked exactly as
    ``answer`` would check it. Rejection is reported by raising, and a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    if isinstance(questions, (str, Mapping)) or not isinstance(questions, Sequence) or not questions:
        raise TypeError("questions must be a non-empty sequence of {question, options} mappings")
    checked = []
    for entry in questions:
        if not isinstance(entry, Mapping) or "question" not in entry or "options" not in entry:
            raise TypeError("each questions entry must be a mapping with 'question' and 'options'")
        _, question, options, _ = _check_inputs(image, entry["question"], entry["options"], max_new_tokens)
        checked.append({"question": question, "options": options, "prompt": format_prompt(question, options)})
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (answer takes one diagram)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "questions": checked,
        "generation": {"max_new_tokens": int(max_new_tokens), "do_sample": False, "decoding": DECODING},
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Sequence[Mapping[str, Any]],
    correct: Sequence[int] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``correct`` (the zero-based index of the correct option per result, in order) the report
    carries the ``accuracy`` (matched option equals the correct index; an unmatched answer counts as
    wrong), the chance baseline, the rate of unmatched answers and one per-question entry, verdict
    ``sample-sanity``; without it the report is ``not-measurable`` and says what labelled data would
    make the task measurable.
    """
    if not results:
        raise ValueError("results must contain at least one answer result")
    base = {
        "task": "diagram image + multiple-choice question -> option text (AI2D-style diagram QA)",
        "score_semantics": (
            "the answer is generated text and carries no score, probability or correctness signal; the "
            "option index is a normalised exact match of that text against the options and is null when "
            "the text matches none of them. Greedy decoding makes the output reproducible on a fixed device "
            "and dtype, a reproducibility property, not a quality one"
        ),
        "sample_kind": sample_kind,
        "n_questions": len(results),
        "truncated": [bool(result.get("truncated")) for result in results],
        "unmatched": [result.get("choice_index") is None for result in results],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if correct is None:
        return {
            **base,
            "metrics": [],
            "baselines": [],
            "verdict": "not-measurable",
            "reason": "no correct-option labels were supplied for the evaluated questions",
            "needs": (
                "multiple-choice question/answer pairs on diagrams from the deployment domain (AI2D-style "
                "annotations with the correct option marked) scored with accuracy; no such labelled set "
                "ships with this repository"
            ),
        }
    if len(correct) != len(results):
        raise ValueError(f"correct has {len(correct)} entries for {len(results)} results")
    per_question = []
    for result, gold in zip(results, correct, strict=True):
        options = list(result.get("options") or [])
        if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(options):
            raise ValueError("each correct entry must be a valid zero-based option index for its result")
        predicted = result.get("choice_index")
        per_question.append(
            {
                "question": result.get("question"),
                "prediction": result.get("answer"),
                "choice_index": predicted,
                "correct_index": gold,
                "correct_option": options[gold],
                "correct": predicted == gold,
            }
        )
    n_options = [len(result.get("options") or []) for result in results]
    chance = sum(1.0 / n for n in n_options) / len(n_options)
    return {
        **base,
        "metrics": [
            {
                "id": "accuracy",
                "value": sum(entry["correct"] for entry in per_question) / len(per_question),
                "normalisation": "answer text lower-cased, punctuation removed, whitespace collapsed; "
                "exact match against the options; unmatched counts as wrong",
                "estimation": f"{len(per_question)} question(s) on one diagram, no dispersion estimate",
            },
            {
                "id": "unmatched_rate",
                "value": sum(entry["choice_index"] is None for entry in per_question) / len(per_question),
                "estimation": f"{len(per_question)} question(s), answers matching no option",
            },
        ],
        "baselines": [{"id": "chance", "value": chance, "note": "mean of 1/n_options over the questions"}],
        "per_question": per_question,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(per_question)} authored question(s) on one tutorial diagram you drew yourself; plumbing "
            "evidence, not an AI2D benchmark"
        ),
        "needs": (
            "a labelled multiple-choice set on diagrams from the deployment domain (subject, drawing style, "
            "label density) for any accuracy claim; the AI2D benchmark is not bundled"
        ),
    }


@dataclass
class Pix2StructAI2DPipeline:
    """``_runner(image, prompt, max_new_tokens)`` returns ``{"answer": str, "new_tokens": int}``."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Pix2StructAI2DPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        font_bytes = header_font_bytes()
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = Pix2StructProcessor.from_pretrained(location, **common)
        if not getattr(processor.image_processor, "is_vqa", False):
            raise RuntimeError("snapshot image processor is not the VQA variant (is_vqa=False); refusing")
        # The checkpoint is stored in bfloat16; it is upcast to float32 for CPU inference.
        model = Pix2StructForConditionalGeneration.from_pretrained(location, dtype=torch.float32, **common)
        model = model.eval().to(resolved_device)

        def runner(image: Image.Image, prompt: str, max_new_tokens: int) -> dict[str, Any]:
            # The image processor is called directly: Pix2StructProcessor.__call__ drops the
            # font_bytes kwarg, and font_bytes is what replaces the default Hub font download
            # (see header_font_bytes). The VQA processor renders the prompt as the header.
            inputs = processor.image_processor(
                image, header_text=prompt, return_tensors="pt", font_bytes=font_bytes
            ).to(resolved_device)
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            # Encoder-decoder: the output holds only decoder tokens (decoder_start + answer + eos).
            answer_ids = generated[0]
            decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return {"answer": decoded, "new_tokens": int(answer_ids.shape[0]) - 1}

        return cls(runner, resolved_device, "float32", source)

    def answer(
        self,
        image: Image.Image,
        question: str,
        options: Sequence[str],
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Answer one multiple-choice question about one diagram; ``answer`` is the decoded text, stripped."""
        rgb, checked_question, checked_options, checked_tokens = _check_inputs(
            image, question, options, max_new_tokens
        )
        prompt = format_prompt(checked_question, checked_options)
        raw = self._runner(rgb, prompt, checked_tokens)
        if not isinstance(raw, dict) or "answer" not in raw:
            raise RuntimeError("runner must return a dict with 'answer'")
        new_tokens = int(raw.get("new_tokens", 0))
        text = str(raw["answer"]).strip()
        return {
            "answer": text,
            "choice_index": match_option(text, checked_options),
            "question": checked_question,
            "options": checked_options,
            "prompt": prompt,
            "image_size": list(rgb.size),
            "new_tokens": new_tokens,
            "truncated": new_tokens >= checked_tokens,
            "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
