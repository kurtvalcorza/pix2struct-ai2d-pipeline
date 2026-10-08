"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings (prefix PSA) fixed in the generator on
top of the fleet-sweep fixes. They need only CI's dependencies: the Section 4 BYOD branch is executed with the
notebook's own source and stand-in inputs (synthetic PNGs, no model), and the rest are static checks on the
generated notebook."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import re
import textwrap
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook_review", TOOLS / "build_notebook.py")
TEMPLATE = _load("notebook_template_review", TOOLS / "notebook_template.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, f"exactly one code cell must contain {marker!r}"
    return found[0]


# --- PSA-m3: the notebook, validator, registry and generator declare spec 2.2 ----------------------------------------


def test_psa_m3_spec_2_2_declared_everywhere(nb: dict) -> None:
    """PSA-m3: metadata, opening cell, registry, validator and generator agree on NOTEBOOK_SPEC 2.2 (§32 item 6)."""
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert build.NOTEBOOK_SPEC == "2.2"
    validator = _load("validate_release_assets_review", TOOLS / "validate_release_assets.py")
    assert validator.NOTEBOOK_SPEC == "2.2"
    assert "DIMER Notebook Specification 2.2 — **standalone** (§4)" in _src(nb["cells"][0])
    assert "DIMER Notebook Specification 2.2" in (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    text = _markdown(nb) + "\n".join(_src(c) for c in _code_cells(nb))
    assert "Specification 2.0" not in text and "NOTEBOOK_SPEC 2.0" not in text


# --- PSA-M3: carried cells titled Infrastructure; a coded experiment that does not run by default -------------------


def test_psa_m3_every_carried_and_install_cell_is_titled_infrastructure(nb: dict) -> None:
    """PSA-M3 acceptance: every carried-module, install and snapshot cell starts with `# @title Infrastructure`, is
    collapsed, and the carried text after the title line still equals the module (PAR1 through embedded_module_text)."""
    titled = 0
    for cell in _code_cells(nb):
        s = _src(cell)
        infra = "# dimer: kernel cell" in s or cell["metadata"].get("dimer", {}).get("embedded_module") or "MANIFEST = {" in s
        if infra:
            assert s.startswith("# @title Infrastructure"), s[:80]
            assert cell["metadata"].get("cellView") == "form"
            titled += 1
    assert titled >= 5
    ctx = build.load_context(ROOT, TEMPLATE, nb["metadata"]["dimer"]["generated_from"]["revision"])
    for cell, module in zip([c for c in _code_cells(nb) if c["metadata"].get("dimer", {}).get("embedded_module")], ctx["modules"], strict=True):
        assert _src(cell).startswith(build.EMBEDDED_TITLE_PREFIX)
        assert build.embedded_module_text(_src(cell)).rstrip("\n") + "\n" == ctx["embedded"][module]
    assert build.embedded_module_text("x = 1\n") == "x = 1\n"


def test_psa_m3_guided_markers_all_present(nb: dict) -> None:
    """PSA-M3 acceptance: the review's static probe markers are all true and at least four cells are form-collapsed."""
    text = "\n".join(_src(c) for c in nb["cells"])
    markers = {
        "how_to_use": r"how to use this notebook", "roadmap": r"roadmap", "glossary": r"glossary", "troubleshooting": r"troubleshoot",
        "infrastructure_label": r"infrastructure", "predict_prompt": r"\bpredict\b(?!ion)|make a prediction",
        "check_your_reasoning": r"check your reasoning|<details", "what_to_notice": r"what to notice|expected result|look for",
        "conclusion_template": r"conclusion template|your conclusion|write.*conclusion",
    }
    missing = [k for k, p in markers.items() if not re.search(p, text, re.I)]
    assert not missing, missing
    assert sum(c.get("metadata", {}).get("cellView") == "form" for c in nb["cells"]) >= 4


def test_psa_m3_coded_experiment_cell_is_off_by_default_and_trains_nothing(nb: dict) -> None:
    """PSA-M3 acceptance / PSA-S1: a coded Predict → change → run → observe experiment exists, is gated off by default,
    and under the default Run all only prints that it was skipped."""
    source = _cell_with(nb, "RUN_OPTION_SHUFFLE_EXPERIMENT = False  # @param")
    assert ".adapt(" not in source and "save_artifact" not in source and "open(" not in source.replace("Image.open(", "")
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<experiment>", "exec"), {})  # no pipe, no records: the default path never needs them
    assert "skipped" in out.getvalue()


# --- PSA-M2: the scored weights are printed; the re-run instructions name the cells -----------------------------------


def test_psa_m2_adapted_flag_is_printed_and_frozen_scores_are_guarded(nb: dict) -> None:
    """PSA-M2 / PSA-S2: Section 6 prints `frozen_test['adapted']` and refuses to label an adapted model's scores
    frozen; Section 8 prints which weights were scored; both run after the SWP-F reload."""
    six = _cell_with(nb, "frozen_test = pipe.evaluate(test_records")
    assert "'adapted': frozen_test['adapted']" in six
    assert "if frozen_test['adapted']:" in six and "not frozen-model numbers" in six
    assert six.index("if pipe.adapter is not None:") < six.index("frozen_test = pipe.evaluate")
    eight = _cell_with(nb, "adapted_test = pipe.evaluate(test_records")
    assert "'adapted_test_adapted': adapted_test['adapted']" in eight
    md = _markdown(nb)
    assert "run the Section 4 cell and every code cell of Sections 5–9 in order" in md
    assert "re-run the Section 7, 8 and 9 cells" in md


# --- PSA-m4: environment-labelled durations ----------------------------------------------------------------------------


def test_psa_m4_prerequisites_state_t4_and_cpu_durations(nb: dict) -> None:
    """PSA-m4: the Prerequisites give the recorded T4 duration and a CPU figure labelled as an estimate."""
    prereq = next(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown" and _src(c).startswith("## Prerequisites"))
    assert "**1,173.5 s**" in prereq and "Tesla T4" in prereq
    assert "CPU has not been timed" in prereq and "not a measurement" in prereq
    assert re.search(r"(minutes|hours?)\b.*CPU|CPU.*\b(minutes|hours?)", prereq)


# --- PSA-m1 / PSA-m5 ---------------------------------------------------------------------------------------------------


def test_psa_m1_no_doubled_braces_in_markdown(nb: dict) -> None:
    """PSA-m1: no `{{`/`}}` reaches the learner; the id pattern renders with single braces."""
    md = _markdown(nb)
    assert "{{" not in md and "}}" not in md
    assert "[A-Za-z0-9_.:-]{1,64}" in md


def test_psa_m5_preinstalled_variable_documented(nb: dict) -> None:
    """PSA-m5: DIMER_NOTEBOOK_CI_PREINSTALLED is explained in markdown wherever code reads it."""
    code = "\n".join(_src(c) for c in _code_cells(nb))
    assert "DIMER_NOTEBOOK_CI_PREINSTALLED" in code
    assert "`DIMER_NOTEBOOK_CI_PREINSTALLED=1` lets an" in _markdown(nb)


# --- PSA-m2: the BYOD branch replayed with the review's inputs ---------------------------------------------------------


def _png(colour: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buf, format="PNG")
    return buf.getvalue()


def _zip(entries: list[tuple[str, bytes | str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in entries:
            z.writestr(n, d)
    return buf.getvalue()


def _records(n: int, image_for=lambda i: f"img{i}.png") -> str:
    return "\n".join(json.dumps({"id": f"q{i}", "image": image_for(i), "question": "What colour?", "options": ["red", "blue"], "answer": i % 2}) for i in range(n))


def _byod_branch(nb: dict) -> str:
    """The `if USE_BYOD:` body of the Section 4 cell, dedented, so it runs with stand-in inputs."""
    source = _cell_with(nb, "BYOD_PATH = ''  # @param")
    body = source.split("\nif USE_BYOD:\n", 1)[1].split("\nelse:\n", 1)[0]
    return textwrap.dedent(body)


def _run_byod(nb: dict, work: Path, payload: bytes | None) -> dict:
    from pix2struct_ai2d_pipeline.samples import MAX_RECORDS, load_byod_dataset, split_dataset

    zip_path = work / "mine.zip"
    if payload is not None:
        zip_path.write_bytes(payload)
    import os
    import shutil

    cwd = Path.cwd()
    os.chdir(work)
    try:
        ns = {
            "Path": Path, "io": io, "zipfile": zipfile, "shutil": shutil, "MAX_RECORDS": MAX_RECORDS, "SPLIT_SEED": 42,
            "load_byod_dataset": load_byod_dataset, "split_dataset": split_dataset, "BYOD_PATH": str(zip_path),
            "byod_file": lambda path, kind, suffixes=(): Path(path),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(_byod_branch(nb), "<byod>", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


def test_psa_m2_byod_valid_zip_is_split(nb: dict, tmp_path: Path) -> None:
    """PSA-m2 (review `byod_replay.valid_16`): a flat zip is accepted and split as before."""
    ns = _run_byod(nb, tmp_path, _zip([("records.jsonl", _records(16))] + [(f"img{i}.png", _png("red" if i % 2 == 0 else "blue")) for i in range(16)]))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 3, "validation": 2, "train": 11}
    assert ns["data_source"].startswith("BYOD (mine.zip)")


def test_psa_m2_byod_nested_folder_paths_are_resolved(nb: dict, tmp_path: Path) -> None:
    """PSA-m2 (review `byod_replay.nested_folder_paths`): folders inside the zip are kept, so `images/img0.png` resolves;
    a records file inside one folder also works."""
    ns = _run_byod(nb, tmp_path, _zip([("records.jsonl", _records(16, lambda i: f"images/img{i}.png"))] + [(f"images/img{i}.png", _png("red")) for i in range(16)]))
    assert sum(len(v) for v in ns["splits"].values()) == 16
    ns = _run_byod(nb, tmp_path, _zip([("pack/records.jsonl", _records(16))] + [(f"pack/img{i}.png", _png("red")) for i in range(16)]))
    assert sum(len(v) for v in ns["splits"].values()) == 16


@pytest.mark.parametrize(
    ("name", "entries", "match"),
    [
        ("no_records_file", [(f"img{i}.png", _png("red")) for i in range(16)], r"mine\.zip: the zip must hold exactly one records\.jsonl.*found none"),
        ("two_records_files", [("records.jsonl", _records(16)), ("sub/records.json", "[]")] + [(f"img{i}.png", _png("red")) for i in range(16)], r"exactly one records\.jsonl.*found \['records\.jsonl', 'sub/records\.json'\]"),
        ("path_traversal", [("records.jsonl", _records(16)), ("../escape.png", _png("red"))] + [(f"img{i}.png", _png("red")) for i in range(16)], r"mine\.zip: member '\.\./escape\.png' points outside the zip"),
        ("answer_out_of_range", [("records.jsonl", _records(16).replace('"answer": 1', '"answer": 5'))] + [(f"img{i}.png", _png("red")) for i in range(16)], r"records\[1\]: answer must be the zero-based index"),
    ],
)
def test_psa_m2_byod_refusals_name_the_zip_and_the_rule(nb: dict, tmp_path: Path, name: str, entries: list, match: str) -> None:
    """PSA-m2 acceptance: each expected failure raises ValueError naming the file and the rule (no StopIteration)."""
    with pytest.raises(ValueError, match=match):
        _run_byod(nb, tmp_path, _zip(entries))


def test_psa_m2_byod_member_count_and_expanded_size_are_capped(nb: dict, tmp_path: Path) -> None:
    """PSA-m2 (§20): a zip with more members than MAX_RECORDS + 2, or more than 2 GiB extracted, is refused before extraction."""
    from pix2struct_ai2d_pipeline.samples import MAX_RECORDS

    too_many = _zip([("records.jsonl", _records(8))] + [(f"f{i}.txt", b"x") for i in range(MAX_RECORDS + 2)])
    with pytest.raises(ValueError, match=rf"mine\.zip: {MAX_RECORDS + 3} files .* exceed the BYOD ceiling of {MAX_RECORDS + 2} files"):
        _run_byod(nb, tmp_path, too_many)
    assert not any((tmp_path / "work" / "byod").glob("f*.txt"))
    branch = _byod_branch(nb)
    assert "BYOD_MAX_EXPANDED_BYTES = MAX_RECORDS + 2, 2 * 1024 ** 3" in branch
    assert "expanded > BYOD_MAX_EXPANDED_BYTES" in branch


def test_psa_m2_cancelled_upload_is_named_not_stopiteration(nb: dict) -> None:
    """PSA-m2 (review `byod_replay.empty_upload_cancelled`): the branch never calls next() on the upload dict."""
    source = _cell_with(nb, "BYOD_PATH = ''  # @param")
    assert "Upload exactly one" in source and "next(iter(uploaded.items()))" in source
    assert source.index("if len(uploaded) != 1:") < source.index("next(iter(uploaded.items()))")
