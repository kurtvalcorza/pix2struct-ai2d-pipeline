"""Multiple-choice diagram dataset contract for fine-tuning: the pinned AI2D sample, validation, seeded
image-disjoint splitting, BYOD loaders and JSONL export.

The default dataset is **real** and from the checkpoint's own domain: questions from the AI2D test split
(Kembhavi et al., ECCV 2016 — school-science diagrams with multiple-choice questions) as mirrored on the
Hugging Face Hub in `lmms-lab-encoder/ai2d`. `google/pix2struct-ai2d-base` was fine-tuned on AI2D's
*training* questions, so this is continued adaptation inside the domain on questions it was not trained on,
not a distribution shift. The sample is one pinned parquet shard (`data/test-00000-of-00002.parquet`,
62,292,686 bytes) downloaded whole at the pinned dataset revision and refused unless its SHA-256 matches the
pin before `pyarrow` reads a byte of it; the diagrams are written to the cache under their own content
digest. A seeded subset of whole diagrams is drawn from it and cut **by diagram** into training,
validation and test questions, so no test diagram is ever trained on.

A record is ``{id, image_id, image, question, options, answer, category}`` — the path of the diagram, the
question, 2..6 distinct options, the zero-based index of the correct option, and a category (`letter-label`
when every option is a one- or two-character diagram label such as `A` or `d`, `text-option` otherwise;
BYOD records may carry any label, `other` by default). Several questions share one diagram; records on the
same `image_id` are always kept in one split.

The shard's SHA-256 and the counts it yields are recorded by `tools/pin_corpus.py` (it needs Hub access);
until they are recorded, `fetch_corpus` refuses to read the shard rather than read an unpinned file.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import (
    MAX_OPTION_CHARS,
    MAX_OPTIONS,
    MAX_QUESTION_CHARS,
    MIN_OPTIONS,
    MODEL_ID,
    normalize_answer,
    validate_image,
)

CORPUS_NAME = "AI2D (test questions)"
CORPUS_REPO = "lmms-lab-encoder/ai2d"
CORPUS_REVISION = "c83a9b9692933aff8349157c88a413df9d02c4e5"
CORPUS_RELEASE = (
    "AI2D test split (3,088 questions) as mirrored on the Hugging Face Hub, dataset revision c83a9b96"
)
CORPUS_LICENSE = (
    "not declared by the Hub mirror; AI2D is published by the Allen Institute for AI "
    "(Kembhavi et al. 2016) — check its terms before redistributing the diagrams or an adapter "
    "trained on them"
)
CORPUS_COLUMNS = ("question", "options", "answer", "image")
# The shard's pins. `sha256`, `rows` and `images` are written by tools/pin_corpus.py from a verified download;
# while `sha256` is None the reader refuses to run.
CORPUS_FILE: dict[str, Any] = {
    "path": "data/test-00000-of-00002.parquet",
    "bytes": 62_292_686,
    "sha256": "450ecfa95b0c475ba214cd9a33b7ec5d1d782e7321a54fc652453d8776743702",
    "rows": 1544,
    "images": 391,
}
DEFAULT_CACHE_DIR = Path("weights") / "ai2d"
SAMPLE_SEED = 42
# Target question counts per split; whole diagrams are allocated until each target is reached, so the
# realised counts can exceed a target by the questions of the last diagram added.
SAMPLE_QUESTIONS = {"train": 360, "validation": 80, "test": 160}
MIN_RECORDS = 8
MAX_RECORDS = 5_000
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
_LABEL_MAX_CHARS = 2


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_pinned() -> bool:
    """Whether the shard's SHA-256 has been recorded (see tools/pin_corpus.py)."""
    return isinstance(CORPUS_FILE.get("sha256"), str) and len(CORPUS_FILE["sha256"]) == 64


def _hub_download(cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            CORPUS_REPO,
            CORPUS_FILE["path"],
            repo_type="dataset",
            revision=CORPUS_REVISION,
            local_dir=str(cache),
        )
    )


def fetch_corpus(
    *, cache_dir: str | Path | None = None, downloader: Callable[[Path], Path] | None = None
) -> Path:
    """Return the path of the pinned shard, downloading it at the pinned revision when the cached copy is
    absent or drifted; refused on any size or SHA-256 mismatch, and outright while no pin is recorded."""
    if not corpus_pinned():
        raise RuntimeError(
            f"{CORPUS_REPO}@{CORPUS_REVISION[:8]} {CORPUS_FILE['path']}: no SHA-256 pin is recorded; run "
            "tools/pin_corpus.py with Hub access to record it before the sample can be read"
        )
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / CORPUS_FILE["path"]

    def ok(path: Path) -> bool:
        return (
            path.is_file()
            and path.stat().st_size == CORPUS_FILE["bytes"]
            and _sha256_file(path) == CORPUS_FILE["sha256"]
        )

    if ok(local):
        return local
    fetched = (downloader or _hub_download)(cache)
    if not ok(fetched):
        size = fetched.stat().st_size if fetched.is_file() else None
        raise ValueError(
            f"{CORPUS_FILE['path']}: fetched {size} bytes, pinned {CORPUS_FILE['bytes']} / "
            f"{CORPUS_FILE['sha256'][:16]}…; refusing to read it"
        )
    return fetched


def read_corpus(path: str | Path) -> list[dict[str, Any]]:
    """The shard's rows as ``{question, options, answer, image_bytes}`` (answer as a zero-based int)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(str(path), columns=list(CORPUS_COLUMNS)).to_pylist()
    out = []
    for index, row in enumerate(rows):
        image = row["image"]
        data = image.get("bytes") if isinstance(image, Mapping) else None
        if not data:
            raise ValueError(f"row {index}: no image bytes")
        out.append(
            {
                "question": str(row["question"]),
                "options": [str(o) for o in row["options"]],
                "answer": int(str(row["answer"]).strip()),
                "image_bytes": bytes(data),
            }
        )
    if CORPUS_FILE.get("rows") is not None and len(out) != CORPUS_FILE["rows"]:
        raise ValueError(f"shard has {len(out)} rows, pinned {CORPUS_FILE['rows']}")
    return out


def option_category(options: Sequence[str]) -> str:
    """`letter-label` when every option is a short diagram label (`A`, `d`, `12`), else `text-option`."""
    return "letter-label" if all(len(str(o).strip()) <= _LABEL_MAX_CHARS for o in options) else "text-option"


def _image_suffix(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as image:
        fmt = (image.format or "PNG").lower()
    return {"jpeg": ".jpg", "png": ".png", "gif": ".gif", "webp": ".webp"}.get(fmt, ".png")


def build_sample_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
    image_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Group the rows by diagram (content digest), shuffle the diagrams with `seed`, and allocate whole
    diagrams to test, validation and train until each split's question target is reached. The diagrams of the
    chosen questions are written to `image_dir` under their digest."""
    sizes = dict(sizes or SAMPLE_QUESTIONS)
    out_dir = Path(image_dir) if image_dir is not None else DEFAULT_CACHE_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_sha256_bytes(row["image_bytes"])[:16], []).append(row)
    if CORPUS_FILE.get("images") is not None and len(groups) != CORPUS_FILE["images"]:
        raise ValueError(f"shard has {len(groups)} distinct diagrams, pinned {CORPUS_FILE['images']}")
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    out: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for image_id in order:
        open_splits = [name for name in ("test", "validation", "train") if len(out[name]) < sizes[name]]
        target = open_splits[0] if open_splits else None
        if target is None:
            break
        data = groups[image_id][0]["image_bytes"]
        path = out_dir / f"{image_id}{_image_suffix(data)}"
        if not path.is_file() or _sha256_bytes(path.read_bytes())[:16] != image_id:
            path.write_bytes(data)
        for row in groups[image_id]:
            out[target].append(
                {
                    "id": f"{target}-{len(out[target]):04d}",
                    "image_id": image_id,
                    "image": str(path),
                    "question": str(row["question"]),
                    "options": [str(o) for o in row["options"]],
                    "answer": int(row["answer"]),
                    "category": option_category(row["options"]),
                }
            )
    short = {name: (len(out[name]), sizes[name]) for name in out if len(out[name]) < sizes[name]}
    if short:
        raise ValueError(f"the shard's diagrams do not fill the split targets: {short}")
    return {"train": out["train"], "validation": out["validation"], "test": out["test"]}


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    downloader: Callable[[Path], Path] | None = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned shard."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    rows = read_corpus(fetch_corpus(cache_dir=cache, downloader=downloader))
    return build_sample_dataset(rows, seed=seed, sizes=sizes, image_dir=cache / "images")


def _check_record(record: Any, index: int, *, base_dir: Path | None) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/question/options/answer")
    for key in ("id", "image", "question", "options", "answer"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    image_ref = record["image"]
    if not isinstance(image_ref, (str, Path)) or not str(image_ref).strip():
        raise ValueError(f"{label}: image must be a file path")
    path = Path(image_ref)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"{label}: image file not found: {path}")
    try:
        with Image.open(path) as handle:
            handle.load()
            validate_image(handle)
            width, height = handle.size
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{label}: image cannot be decoded: {exc}") from exc
    question = record["question"]
    if not isinstance(question, str) or not " ".join(question.split()):
        raise ValueError(f"{label}: question must be a non-empty string")
    question = " ".join(question.split())
    if len(question) > MAX_QUESTION_CHARS:
        raise ValueError(f"{label}: question exceeds MAX_QUESTION_CHARS={MAX_QUESTION_CHARS}")
    options = record["options"]
    if isinstance(options, str) or not isinstance(options, Sequence):
        raise ValueError(f"{label}: options must be a list of strings")
    if not MIN_OPTIONS <= len(options) <= MAX_OPTIONS:
        raise ValueError(f"{label}: options must have {MIN_OPTIONS}..{MAX_OPTIONS} entries")
    checked_options = [" ".join(str(o).split()) if isinstance(o, str) else "" for o in options]
    if not all(checked_options):
        raise ValueError(f"{label}: every option must be a non-empty string")
    if any(len(o) > MAX_OPTION_CHARS for o in checked_options):
        raise ValueError(f"{label}: an option exceeds MAX_OPTION_CHARS={MAX_OPTION_CHARS}")
    if len({normalize_answer(o) for o in checked_options}) != len(checked_options):
        raise ValueError(f"{label}: options must be distinct after normalisation")
    answer = record["answer"]
    if isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(checked_options):
        raise ValueError(f"{label}: answer must be the zero-based index of the correct option")
    return {
        "id": rid,
        "image_id": str(record.get("image_id", path.name)),
        "image": str(path),
        "image_size": [width, height],
        "question": question,
        "options": checked_options,
        "answer": answer,
        "category": str(record.get("category", "other")),
    }


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Structural validation of a multiple-choice dataset (every diagram opened and decoded, every question
    held to the same ceilings `answer` applies); raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, image, question, options, answer} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    base = Path(base_dir) if base_dir is not None else None
    checked = []
    ids: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index, base_dir=base)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_images": len({r["image_id"] for r in checked}),
        "categories": dict(Counter(r["category"] for r in checked)),
        "options_per_question": {
            "min": min(len(r["options"]) for r in checked),
            "max": max(len(r["options"]) for r in checked),
        },
        "answer_positions": dict(sorted(Counter(r["answer"] for r in checked).items())),
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [r["id"], r["image_id"], r["question"], list(r["options"]), int(r["answer"]), r.get("category", "")]
        for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no diagram appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = str(record.get("image_id", record["id"]))
            if key in seen and seen[key] != name:
                raise ValueError(f"diagram {key!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
    base_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded split of a BYOD dataset into train/validation/test **by diagram**: every question on the same
    diagram lands in the same split, so a test diagram is never seen in training."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records, base_dir=base_dir)["records"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        groups.setdefault(record["image_id"], []).append(record)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    n_test = max(1, round(len(checked) * test_fraction))
    n_val = round(len(checked) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for group in order:
        if len(splits["test"]) < n_test:
            splits["test"].extend(group)
        elif len(splits["validation"]) < n_val:
            splits["validation"].extend(group)
        else:
            splits["train"].extend(group)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSON array or a JSONL file of ``{id, image, question, options, answer}`` objects;
    `image` paths are resolved relative to the file's directory by `validate_dataset(..., base_dir=...)`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .json or .jsonl")


def write_dataset_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One record per line in the shape `load_byod_dataset` reads back (image paths as given)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ("id", "image_id", "image", "question", "options", "answer", "category")
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({k: record[k] for k in keys if k in record}, ensure_ascii=False) + "\n")
    return out
