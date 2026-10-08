# Notebook review: `tutorials/pix2struct_ai2d_colab.ipynb`

Notebook Review Framework v1 review against DIMER Notebook Specification 2.2. Review only: nothing in the
repository was changed. Finding prefix: **PSA**.

## 1. Scope and evidence

### Review contract

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/pix2struct-ai2d-pipeline` |
| Notebook | `tutorials/pix2struct_ai2d_colab.ipynb` |
| Reviewed revision | `origin/main` = `6dce21461b0796528934b8ea93c8cb7ef5fccd90` (notebook blob `02cd3fb8fda4f021d1755fdb0b062f1952c70fbe`) |
| Generated from | `metadata.dimer.generated_from.revision` `a633e1abe215…`, `tools/build_notebook.py` / `tools/notebook_template.py`; `build_notebook.py --check` reports the notebook up to date |
| Requirements baseline | NOTEBOOK_SPEC **2.2** (ml-worker `origin/main`). The notebook declares **2.0** |
| Profile / mode | `E2E` / `GUIDED` (metadata and opening cell) |
| Intended audience | Stated as knowing basic Python and PIL, encoder–decoder generated tokens, and accuracy against chance and baselines (Prerequisites) |
| Supported runtime | Colab or Jupyter, Python 3.12. CPU float32 by default, CUDA used when present, GPU "recommended for Sections 6–8" |
| Promised outcomes | Stage and digest-verify the pinned snapshot; fetch the digest-pinned AI2D shard and split it by diagram; inference contract on a drawn diagram; frozen accuracy beside chance and two baselines; bounded decoder fine-tuning with validation epoch selection; held-out evaluation per category; safetensors adapter export with reload parity; optional BYOD through the same stages |

### Existing execution evidence

`docs/release-verification.md` records a **PASSED** Kaggle Tesla T4 run of commit `75255e0` / blob `02cd3fb8fda4`.
That blob is the one under review, so the record covers this revision. The run was "11/11 code cells ok
**after the install restart**". Pass 1 stopped at the install cell with the notebook's stale-module guard
(`cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3). The kernel was restarted and pass 2 completed. No
Colab run of this blob is recorded. No BYOD run and no run of an optional experiment are recorded.

### Journeys and evidence basis

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection of all 25 cells | Barriers found: PSA-M3, PSA-m1, PSA-m4 |
| Clean default | Documented execution evidence (Kaggle T4, exact blob). Not run here: there is no local GPU by workspace rule, and the 565 MB checkpoint was not fetched | Completes only after a manual restart (PSA-M1). **Colab not verified** |
| Active learning | Direct execution at **reduced scale**: the repository's own 3-layer, 32-wide random Pix2Struct on CPU, calling `adapt` twice as the "Optional experiments" rerun does | Fails: adaptation stacks on the already-adapted model, and the artifact stops reproducing the in-memory model (PSA-M2) |
| Reuse and recovery | Direct execution of the BYOD branch's verbatim upload/extract/load/split logic with a stubbed upload (no model). Artifact reload: documented evidence (8/8) | A valid zip is accepted and split. One invalid input gets an actionable message. Two get a bare `StopIteration`, and a nested-folder zip gets a misleading one (PSA-m2). BYOD downstream stages with the real model: **not verified** |

Limitations: no learner observation. No Colab execution. The full-scale numbers are taken from the repository
record and were not reproduced. Repository checks run here are source checks, not execution evidence (REL8):
`build_notebook.py --check` exited 0, `validate_release_assets.py` reported PASS, and
`pytest tests/test_notebook_parity.py tests/test_adaptation_model.py` gave 11 passed and 7 skipped (the
real-checkpoint and CUDA cases skip).

### Promise → evidence trace

| Claim | Implementation | Observable result | Learner interpretation |
|---|---|---|---|
| Run all completes with no intervention (opening cell, "NOTEBOOK_SPEC 2.0 §5") | Cell 3 pins install plus stale-module guard | Kaggle T4: `RuntimeError` asking for a restart, then pass 2 OK | Contradicted by the documented run (PSA-M1) |
| Pinned, digest-verified snapshot | Cell 11 `stage_missing_files` + `verify_snapshot` | Documented: 8 files, sha `652c92f8…` | Delivered |
| Diagram-disjoint split of a pinned shard | Cell 13 `fetch_corpus` / `build_sample_dataset` / `check_split_disjoint` | Documented: 360/82/161 on 87/24/41 diagrams, 4 refusal probes | Delivered |
| Frozen vs chance vs baselines | Cell 17 | Documented: 0.354 vs 0.25 / 0.193 / 0.248 | Delivered and explained (Section 6 markdown) |
| Bounded fine-tuning, validation selection | Cell 19 `pipe.adapt` | Documented: best epoch 2, 18,879,744 trainable | Delivered on the default path. Breaks on rerun (PSA-M2) |
| Held-out comparison per category | Cell 21 | Documented: adapted 0.366, unmatched 0.068 | Delivered. The interpretation text is careful about the unmatched-rate confound |
| Adapter export and fresh reload | Cell 23 `save_artifact` / `from_artifact` + assert | Documented: 8/8 parity | Delivered on the default path. Not guaranteed after a rerun (PSA-M2) |
| "Optional experiments … do not affect the default path" | Prose only (cell 24) | Rerunning Section 7 stacks adaptation (probe) | Misleading (PSA-M2) |
| BYOD through the same stages | Cell 13 `USE_BYOD` branch | Valid zip reaches split. Re-running "from that cell" scores an already-adapted model as "frozen" | Partly delivered (PSA-M2, PSA-m2) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Read `answer` / `choice_index` / `truncated` correctly | Printed answers in Section 5 | Output is shown. No question asks the learner to read it (PSA-M3) |
| Read accuracy against chance and baselines, and the category breakdown | Section 6 and 8 prints plus the "read it in this order" prose | Guidance is good. No checkpoint or prediction (PSA-M3) |
| Run a bounded fine-tuning with explicit hyperparameters | Form fields in Section 7 | Exercised on the default path. Changing a field and rerunning gives invalid results (PSA-M2) |
| Export an adapter that reloads with parity | Section 9 | Exercised (assert) |

## 2. Separate judgments

- **Technical correctness:** the default path is correct and well guarded: digest pins, transactional `adapt`,
  pre-deserialisation artifact checks. Two defects. The documented install restart is a `MUST` failure. `adapt`
  mutates the shared `pipe` and starts from its current weights, so any rerun gives invalid results.
- **Promise fulfilment:** the default-path promises are met by documented evidence, except the "Run all with no
  intervention" promise. The optional-experiment and BYOD-rerun promises are not met as written.
- **Learner experience:** the explanation is strong and honest. The baseline framing, the unmatched-rate confound
  and the in-domain caveat are all stated clearly. But the GUIDED learner meets about 67,000 characters of
  unlabelled carried module code before any model runs. There is no How-to-use section, roadmap, glossary,
  troubleshooting, prediction, checkpoint or conclusion scaffold.
- **Spec conformance:** fails the `MUST`s RUN1, RUN10 and ENV6 (restart). It declares spec 2.0 rather than 2.2.
  GDL1–GDL14 and EXE2/EXE5 `SHOULD` deviations are not recorded as deviations. All other applicable `MUST`s
  checked by source inspection appear met: ST1–ST8, MOD1–MOD9, DAT1–DAT9, VAL1–VAL8, SPL, FT, EVAL, UNC, ART and VER
  on the default path.

## 3. Findings

### Major

**PSA-M1 — Section 1 install cell: the default `Run all` requires a manual restart.**
Cell 3 `pip install`s exact pins (`numpy==2.5.3`, `torch==2.14.0`, …) into the running kernel. If a pinned
distribution was already imported, it raises "Restart the runtime, then rerun from the top". The only recorded
clean run of this blob hit exactly that: Kaggle T4 pass 1 stopped at the install cell (`cuda-bindings`, `numpy`),
and the kernel was restarted before pass 2 completed. `docs/release-verification.md` step 4 calls the restart
"expected" and still records the notebook as Release-grade. `tutorials/README.md` marks Run-all "verified".
- *Consequence:* a learner choosing Run all on a fresh hosted runtime stops at cell 3. A Run all that needs a
  restart is not conformant (§5 closing rule, §25.7), so the Release-grade label overstates the evidence.
- *Evidence:* documented execution evidence (release-verification.md, 2026-09-26 row: "11/11 code cells ok after
  the install restart", "pass 1 178.1 s stopped at the install cell"). Source inspection of cell 3 and
  `tools/build_notebook.py:61–69`. Colab: not verified.
- *Recommended correction:* replace the kernel install with the fleet's uv isolated-environment pattern. A carrier
  cell bootstraps uv, creates `uv venv --managed-python --python 3.12.12 <ROOT>/env`, installs a hash-locked
  `requirements.txt` with `uv pip install --require-hashes --only-binary :all:`, and runs the workload in that
  env, so the kernel's preloaded NumPy/torch are never replaced. Reference:
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` on origin/main.
  Change it in `tools/build_notebook.py` (install cell, lines ~445–470), not by hand. Then correct
  `docs/release-verification.md` step 4 and the registry status until a no-restart run exists.
- *Acceptance check:* a fresh Colab (or Kaggle) runtime executes every code cell in order in one kernel, with no
  restart and no error output, recorded against the new blob. `docs/release-verification.md` no longer describes a
  restart as expected.
- *Spec:* RUN1, RUN10, ENV6 (MUST); REL2, REL11.

**PSA-M2 — Sections 4, 6, 7, 9: reruns adapt an already-adapted model, so the optional experiments and the BYOD
rerun give invalid comparisons and a non-reproducing artifact.**
`pipe.adapt` (carried `pipeline.py`, `adapt`) trains in place, starting from the pipeline's *current* weights,
and still labels epoch 0 "frozen model". Nothing in the notebook restores the base before Section 6 or 7.
- The opening and cell 24 tell the learner to "raise `LEARNING_RATE`" or "set `TRAINABLE_DECODER_LAYERS = 1`" and
  compare, adding "they do not affect the default path". Rerunning Section 7 continues training the adapted
  model, and its "frozen" epoch 0 is the previous adapter.
- The BYOD instruction says "set `USE_BYOD = True` in Section 4 and re-run from that cell". Section 6 then scores
  the **AI2D-adapted** model as the "frozen model" on the user's test split. Section 7 adapts on top of it, and the
  exported artifact silently contains AI2D adaptation while its metadata names only the BYOD source.
- With `TRAINABLE_DECODER_LAYERS = 1` after the default 2, the earlier-trained block stays modified in memory but
  is not exported, so the artifact no longer reproduces the in-memory model. The 8-question text-parity assert may
  or may not catch this.
- *Consequence:* the learner's "frozen vs adapted" and "1 vs 2 blocks" conclusions are wrong. A BYOD user
  attributes AI2D-adapted behaviour to the base model. The exported adapter misdescribes its own lineage. This is
  framework dimension 3 (validity) and dimension 7 (an exercise leaves the notebook in an inconsistent state).
- *Evidence:* direct execution, **reduced scale** (`run_probes.py`, `stacked_adaptation`): 3-layer random model,
  CPU, no validation split so the weights move. After `adapt(layers=2)` then `adapt(layers=1)`, the second run did
  not start from base (`run2_started_from_base: false`) and still reported epoch 0 as `"frozen model"`. All 14
  `decoder.layer.1.*` tensors differ from base, and an artifact reloaded onto a fresh base mismatches the in-memory
  model on 14 tensors. Source inspection of cell 13, cell 24 and `pipeline.py` `adapt`.
- *Recommended correction:* make each experiment start from the verified base. Either capture the base
  state of the adaptable tensors right after `from_pretrained` and restore it at the top of Sections 6 and 7, or
  rebuild `pipe` with `from_pretrained(weights_dir=WEIGHTS_DIR)` there (no download, the snapshot is already
  verified). Also print `frozen_test['adapted']` in Section 6. Correct the BYOD and optional-experiment rerun
  instructions to name the exact cells to rerun. Change `tools/notebook_template.py` around lines 328, 369 and 520,
  and line 39 for the BYOD instruction.
- *Acceptance check:* with the default path completed, rerunning Section 7 with `TRAINABLE_DECODER_LAYERS = 1`
  produces an epoch 0 identical to the Section 6 frozen accuracy, and every non-exported decoder tensor equals
  base. Re-running from Section 4 prints `adapted: False` for the Section 6 evaluation. A reduced-scale probe like
  `stacked_adaptation` reports 0 mismatched tensors.
- *Spec:* GDL10, UX7/RUN9 (exercises must not corrupt state), DAT13/DAT14 (BYOD same semantics), VER3–VER5, ART8.

**PSA-M3 — Whole notebook: the declared `GUIDED` layer is largely missing, and about 67,000 characters of carried
code are unlabelled.**
Static probe results: no "How to use this notebook", roadmap, glossary, troubleshooting, collapsible sample
answers or conclusion scaffold; 0 cells with `cellView: form`; no "Infrastructure" label. The three carried
module cells hold 4,537, 43,537 and 19,098 characters and sit between Section 1 and the first model step. The
learner activity is limited to "Optional experiments" prose with no code and no predict/explain prompts. The
notebook never asks the learner to state a prediction or interpret an output.
- *Consequence:* the stated learner can follow the default path but is never asked to apply the stated
  objectives, such as reading `choice_index`, reading the baselines or explaining the category split. The long
  infrastructure cells look like prerequisite ML knowledge (framework dimensions 4 and 6). Severity is set by
  learner consequence. Spec conformance alone is `SHOULD`.
- *Evidence:* source inspection plus a static probe (`results.json` `static.guided_markers`,
  `cellView_form_cells: 0`).
- *Recommended correction:* in `tools/build_notebook.py` / `tools/notebook_template.py`, add How-to-use, a roadmap
  and an Input → Model → Output line to the opening. Title the carried-module, manifest and install cells
  `# @title Infrastructure: …` with `cellView: form` (parity tests must account for the title line). Add a
  prediction before Sections 6 and 8, a collapsible "Check your reasoning" after them, one coded Predict → Change one
  thing → Run → Observe → Explain activity (built on the PSA-M2 fix), a troubleshooting section (no GPU and slow
  CPU, download or digest failure, OOM, BYOD rejections) and a conclusion template. Follow the 2.2 reference
  notebook named in spec §25.13.
- *Acceptance check:* the static probe reports all `guided_markers` true, `cellView_form_cells` ≥ 4, and every
  carried or install cell titled "Infrastructure". At least one coded experiment cell exists that does not run under
  default Run all.
- *Spec:* GDL1–GDL14, UX5, UX8 (SHOULD).

### Minor

**PSA-m1 — Opening cell and Prerequisites: doubled braces from template escaping appear in the learner text.**
The rendered markdown shows `{{id, image, question, options, answer}}` (cells 0 and 1) and the id pattern
`[A-Za-z0-9_.:-]{{1,64}}` (cell 1). The real pattern is `{1,64}`, so a learner copying it gets a wrong regex.
*Evidence:* static probe `double_brace_in_markdown` (3 hits). Source `tools/notebook_template.py:40,127`.
*Correction:* stop formatting these strings, or write single braces. *Acceptance:* no `{{`/`}}` in any markdown
cell. *Spec:* SRC3 (knowingly stale text), UX2.

**PSA-m2 — Section 4 BYOD branch: some expected failures give no actionable message, there is no location field,
and there is no size limit.**
A cancelled upload or a zip without `records.jsonl`/`records.json` raises a bare `StopIteration` with an empty
message. A zip whose records use folder paths (`images/img0.png`), as most users would pack it, is flattened on
extraction and then rejected as "image file not found: …/byod/images/img0.png". That message names a path the
notebook itself removed, without saying that folders are flattened. The branch always opens `files.upload()` and
has no location field. The zip has no member-count or expanded-size ceiling. *Evidence:* direct execution of the
branch logic (`byod_replay`: `no_records_file` and `empty_upload_cancelled` → `StopIteration`; `nested_folder_paths`
→ misleading path; `answer_out_of_range` → actionable). *Correction:* explicit checks with messages that name the
missing `records.jsonl`, the empty upload and the flattening rule (or resolve records by basename); a
`BYOD_ZIP_PATH = ''  # @param` field that bypasses the upload when set; an expanded-size cap. Change these in
`tools/notebook_template.py:159–180`. *Acceptance:* each of the three inputs raises a `ValueError` naming the
failed rule and the fix, and setting `BYOD_ZIP_PATH` runs without importing `google.colab`. *Spec:* DAT19, UX10,
EXE1, EXE2, §20 (expanded-size SHOULD).

**PSA-m3 — Metadata, opening and README: the notebook declares spec 2.0 where the fleet baseline is 2.2.**
*Evidence:* `metadata.dimer.notebook_spec = "2.0"`; the opening cell, `tutorials/README.md` and the validator
all say 2.0. *Correction:* migrate to 2.2 together with PSA-M3, and update the validator in the same change
(§32 item 6). *Acceptance:* metadata, opening cell, registry and validator agree on 2.2. *Spec:* §3.4, §32.

**PSA-m4 — Prerequisites: no runtime estimate for the CPU path the notebook says works.**
The Colab badge opens a CPU runtime by default. The notebook says "the CPU path works but is slow" and points to
`docs/release-verification.md` for timings, which records only a T4 run (1,173.5 s total, adaptation 688.6 s). A
CPU learner has no estimate for a path that encodes about 1,800 question-diagram composites (estimate from the default split sizes and three epochs) at up to 2,048
patches. *Evidence:* source inspection. The CPU duration was not measured here. *Correction:* state the T4 timing
in the notebook, give a measured or labelled-estimate CPU figure, and recommend a GPU runtime before Section 1.
*Acceptance:* the Prerequisites state an environment-labelled duration for T4 and for CPU. *Spec:* RUN12, UX12,
GDL1.

**PSA-m5 — Section 1: the environment variable `DIMER_NOTEBOOK_CI_PREINSTALLED` is read but never documented.**
Setting it skips the install, so a reader may not know why versions differ. *Evidence:* static probe
(`env_var_documented_in_markdown: false`). *Correction:* one sentence in the Section 1 markdown. *Acceptance:*
the variable is named and explained in markdown. *Spec:* EXE5.

### Suggestions

- **PSA-S1 — cell 24 optional experiments.** Provide code for "shuffle the options of a test question and watch the
  answer follow them". It is the most instructive experiment and today needs the learner to write code.
- **PSA-S2 — Sections 6 and 8.** Print the `adapted` flag that `pipe.evaluate` already returns, so the learner can
  see which weights were scored.
- **PSA-S3 — fast path.** Offer a documented reduced-scale option, for example smaller `SAMPLE_QUESTIONS` or one
  epoch, for CPU-only learners (GDL3).
- **PSA-S4 — References.** Give DOIs/APA entries for Lee et al. (2023) and Kembhavi et al. (2016), matching the
  2.2 reference notebook's scholarly-grounding pattern (UX2, SRC10).

## 4. Readiness

**Needs revision.** Open Majors: PSA-M1 (a documented manual restart, which fails `MUST`s RUN1, RUN10 and ENV6),
PSA-M2 (reruns invalidate the experiments, the BYOD comparison and the artifact lineage) and PSA-M3 (guided layer).
The default-path stages, the evaluation design and the artifact contract are otherwise sound, and are evidenced by
the exact-blob Kaggle T4 run. Remaining gates after the fixes: a no-restart clean run of the new blob on
Colab or Kaggle, a recorded BYOD positive and negative run with the real model (REL12), and the 2.2 migration.

## 5. Verified vs inferred

- **Verified here (direct execution, CPU, reduced scale):** BYOD branch logic on 5 inputs; stacked adaptation and
  artifact mismatch on the repository's tiny random Pix2Struct; generator `--check`, release-asset validator,
  parity and tiny-model tests (static/unit, not execution evidence).
- **From documented evidence:** the default-path numbers and the restart (Kaggle T4, exact blob).
- **Inferred:** that Colab also trips the restart guard (Kaggle did; Colab is not verified); that PSA-M2 changes
  real-model answers at full scale (shown on tensors at reduced scale); CPU duration.
- **Most likely wrong:** the severity of PSA-M3. The guided layer is `SHOULD`, and spec 2.2 says existing 2.x
  notebooks do not become nonconformant solely for lacking it. A reviewer weighting spec conformance over learner
  activity would grade it Minor.
