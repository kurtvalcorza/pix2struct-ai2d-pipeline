# Release verification

`tutorials/pix2struct_ai2d_colab.ipynb` (`E2E`, **standalone** carrier) is **Candidate**. Release-grade applies only to
the exact commit and notebook blob a release review promotes against "Recorded executions" below, and any change to
the blob returns the carrier to **Candidate** until that exact blob has executed top-to-bottom in a clean supported
runtime and is reviewed. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the single kernel cell that builds (or reuses, by lock digest) the isolated hash-locked uv environment and routes every later cell to it, with no `pip install` into the kernel and no restart request; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned
  AI2D dataset revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Pix2StructAI2DPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` and `read_corpus` from the
  pinned cache path, `build_sample_dataset(..., seed=SPLIT_SEED, image_dir=...)` / `load_byod_dataset`,
  `validate_dataset` per split, `check_split_disjoint`, `write_dataset_jsonl`, the ceiling print, `validate_inputs`
  with the single-option refusal probe, `pipe.answer` with the sanity checks, `evaluation_report` on the drawn
  diagram, `position_prior_baseline`, `longest_option_baseline`, `pipe.evaluate` on the frozen model and on the
  validation and test splits after adaptation with the per-category breakdown, `pipe.adapt` with its explicit
  hyperparameters, `evaluation_report` on the diagram after adaptation, `pipe.save_artifact`,
  `Pix2StructAI2DPipeline.from_artifact` and the reload-parity assertion, the recorded `frozen_beats_chance` and
  `adapted_beats_frozen` flags, and the provenance fields `weight_format`, `weight_sha256` and the
  `corpus` block), the six expected `outputs/` paths, the learner-facing statements and the gated-off BYOD default;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a
  mutable `revision='main'`, direct `from transformers import` / `Pix2StructForConditionalGeneration` /
  `Pix2StructProcessor` / `.generate(` / `from huggingface_hub import` / `urllib.request` / `pyarrow` /
  `safetensors` / `torch.optim` / `.backward(` / `pipe._model` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy`, `pillow`,
`huggingface-hub` and `pyarrow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
offline suites (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner and shard downloader,
tiny PIL drawings, a small generated parquet shard, temporary manifests, no weights).
`tests/test_adaptation_model.py` builds a 3-layer, 32-wide random Pix2Struct from the committed config, VQA
processor and tokenizer (header rendered in Pillow's bundled font) and runs the real `evaluate` / `adapt` /
`save_artifact` / `load_artifact` path on it offline; its pinned-checkpoint and CUDA cases skip unless
`model.safetensors` is staged and a GPU is visible. These are source/provenance and unit checks. They are **not**
execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; a GPU runtime is recommended for Sections 6–8) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/pix2struct-ai2d-base/` or the corpus cache `weights/ai2d/`, and with the shard's SHA-256 pin
   recorded in the committed `samples.py` (the reader refuses to run otherwise);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `ANSWER_MAX_TOKENS = 16`, `EPOCHS = 3`, `LEARNING_RATE = 1e-5`,
   `BATCH_SIZE = 4`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`; they are installed into the isolated environment Section 1 builds, so no interpreter restart is
   expected);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `google/pix2struct-ai2d-base` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files), and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with no further Hub access (a
     font fetch in the logs after staging is a finding — the header font is Pillow's bundled one);
   - Section 4: `fetch_corpus` downloading `data/test-00000-of-00002.parquet` at the pinned dataset revision and
     accepting it only with the pinned size and SHA-256; `read_corpus` returning the pinned row count; the seeded
     allocation of whole diagrams into about 360 / 80 / 160 training, validation and test questions with
     `check_split_disjoint` reporting no shared diagram, the category mix and correct-position counts printed and
     the three dataset digests; `outputs/pix2struct_ai2d_train.jsonl` written; the four dataset refusal probes each
     raising `ValueError`;
   - Section 5: the ceilings surfaced; the plant diagram drawn; `validate_inputs` writing
     `outputs/pix2struct_ai2d_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the
     single-option probe); `pipe.answer` on the ten questions with every sanity check `True` and
     `evaluation_report` verdict `sample-sanity` (the inference-only notebook's runs scored 4/10 against chance 0.25;
     a different answer on another runtime is a finding to record, not a failure);
   - Section 6: chance, the position-prior and longest-option baselines and the frozen model's test accuracy,
     unmatched rate and per-category accuracy, and `frozen_beats_chance` printed;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 18,879,744 trainable of 282,285,696
     parameters (29 tensors: two decoder blocks and the final layer norm), the training-question count, and the
     epoch history with validation accuracy;
   - Section 8: `pipe.evaluate` on the validation and test splits with the five-way comparison, the per-category
     breakdown, `adapted_beats_frozen` printed and `outputs/pix2struct_ai2d_evaluation_report.json` written (the
     gain is **recorded, not asserted**, until a measured recipe is on file);
   - Section 9: the ten questions answered by the adapted model with the `sample-sanity` report,
     `outputs/pix2struct_ai2d_answers.csv` written; `pipe.save_artifact` writing
     `outputs/pix2struct_ai2d_adapter/{adapter.safetensors,manifest.json}` and
     `Pix2StructAI2DPipeline.from_artifact` reloading it with identical answers on eight test questions (the cell
     asserts it); `outputs/pix2struct_ai2d_result.json` written with `NOTEBOOK_SOURCE`, the model identity and
     licence, the snapshot block, the `corpus` block, the inference-contract items, the comparison, the artifact
     digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark), the value of
   `frozen_beats_chance` and `adapted_beats_frozen` and any warning or applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Recorded executions

Notebook identity is the Git blob id of `tutorials/pix2struct_ai2d_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/pix2struct_ai2d_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### `E2E` notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-08 (00:31–00:48) | `39d8e03` / `bddb37243376` (`NOTEBOOK_SOURCE.repository_revision` = `a89788a…`, the source revision the generator embedded; `a89788a..39d8e03` changes only the isolated worker's `google.colab` stubs, which now carry a module spec, plus their regression test; embedded `module_sha256` `2c028c60827d…`, generator `build_notebook.py/2.2`) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM, committed blob fetched at the 40-char SHA and Git-blob verified; kernel Python 3.13.15; isolated uv environment built from the 48-package hash lock (Python 3.12.12, 57 s), torch 2.14.0+cu130, transformers 4.57.6, device `cuda:0`, float32, source `local-snapshot` | Default path, form parameters at their defaults, 13/13 code cells in order (cells 3–5 are the carried module definitions and print nothing). Staging: 8 manifest files (568,740,298 B) from `google/pix2struct-ai2d-base` @ `0d6b2606…`, `verify_snapshot` 8 files. Section 4: shard sha256 `450ecfa95b0c…` (1,544 rows, 391 diagrams), split 360 / 82 / 161 questions on 87 / 24 / 41 diagrams, digests `f1fea8d53bb9…` / `57f7558734ce…` / `9968c8d6d3a3…`, four refusal probes rejected. Section 5: manifest `accepted` (1 finding), ten answers `root`, `root`, `root`, `stem`, `sun`, `soil`, `root`, `sun`, `leaf`, `root`, `sample-sanity` accuracy 0.4 vs chance 0.25, unmatched 0. Section 6 (n = 161): chance 0.25, position prior 0.193, longest option 0.248, frozen 0.354 (unmatched 0.075; letter-label 0.343 of 35, text-option 0.357 of 126). Section 7: 18,879,744 of 282,285,696 parameters trainable; validation accuracy 0.366 → 0.366 → 0.378 → 0.354 (loss 1.7402 / 1.5653 / 1.3107), best epoch 2; 655.2 s. Section 8: test adapted 0.366 (unmatched 0.068; letter-label 0.314, text-option 0.381), delta +0.012 accuracy, −0.006 unmatched, `adapted_beats_frozen` `True` (recorded, not asserted). Section 9: drawn diagram 0.4 / 0.4; adapter 29 tensors, 75,522,608 B, sha256 `e6b175146c25…`; reload parity 8/8. Not exercised: BYOD upload, the optional option-shuffle experiment (skipped by default) and the other activities. Byte-exact evidence in `docs/execution-evidence/2026-10-08-39d8e03/`: executed notebook sha256 `03c764e091c8…`, `exec.log` `d49f9302106b…`, `run_summary.json` `4bcbdd05b5da…` | 1029.1 s (one pass, no restart) | **PASSED** — one pass, no restart, 0 errors; evidence for this blob; status stays Candidate pending review |
| 2026-09-26 (03:30–03:50) | `75255e0` / `02cd3fb8fda4` (`NOTEBOOK_SOURCE.repository_revision` = `metadata.dimer.generated_from.revision` = `a633e1a…`, the source revision the notebook was generated at; `a633e1a..75255e0` changes only the notebook; embedded `module_sha256` `2c028c60827d…`) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pix2struct-ai2d` v3), serial suite: committed blob fetched at the 40-char SHA and Git-blob verified, executed verbatim in a fresh interpreter with a `google.colab` shim and no repository checkout; HF cache clean at start, no pre-staged snapshot or corpus files; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461` (image torch 2.10.0+cu128, transformers 5.0.0, numpy 2.0.2, pillow 11.3.0), Python 3.12.13, Tesla T4 15360 MiB, driver 580.159.04; after the inline pins: torch 2.14.0+cu130 (CUDA 13.0), transformers 4.57.6, device `cuda:0`, float32, source `local-snapshot` | Default path, form parameters at their defaults (`USE_BYOD = False`, `SPLIT_SEED = 42`, `ANSWER_MAX_TOKENS = 16`, `EPOCHS = 3`, `LEARNING_RATE = 1e-5`, `BATCH_SIZE = 4`, `TRAINABLE_DECODER_LAYERS = 2`), 11/11 code cells ok after the install restart. Staging: all 8 manifest entries fetched from `google/pix2struct-ai2d-base` @ `0d6b2606…` (568,740,298 B), `verify_snapshot` 8 files, `model.safetensors` sha256 `652c92f8b995…`; no font fetch in the log. Section 4: `data/test-00000-of-00002.parquet` @ `c83a9b96…` accepted at the pinned 62,292,686 B / sha256 `450ecfa95b0c…` (1,544 rows, 391 diagrams); split 360 / 82 / 161 questions on 87 / 24 / 41 whole diagrams (train 300 text-option + 60 letter-label; test 126 + 35), dataset digests `f1fea8d53bb9…` / `57f7558734ce…` / `9968c8d6d3a3…`; the four refusal probes (duplicate id, missing image, answer out of range, too small) each rejected. Section 5: input manifest `accepted` with the single-option rejection finding; ten answers `root`, `root`, `root`, `stem`, `sun`, `soil`, `root`, `sun`, `leaf`, `root`, every sanity check `True`, `sample-sanity` `accuracy` 0.4 (4/10) against chance 0.25, `unmatched_rate` 0.0 (finding: labels 4 and 6 answered `stem` / `soil` here where the 2026-09-14 CPU runs answered `root` / `water`; the score is unchanged). Section 6 (test n = 161): chance 0.25, position-prior 0.193, longest-option 0.248, frozen accuracy 0.354 (unmatched 0.075; letter-label 0.343 of 35, text-option 0.357 of 126), `frozen_beats_chance` `True`. Section 7: 18,879,744 trainable of 282,285,696 parameters, 360 training questions, seed 0; validation accuracy epoch 0 (frozen) 0.366, epoch 1 0.366 (loss 1.7402), epoch 2 0.378 (loss 1.5653), epoch 3 0.354 (loss 1.3107); best epoch 2 by validation accuracy; 688.6 s. Section 8: validation (n = 82) adapted 0.378, unmatched 0.049; test adapted accuracy 0.366 (unmatched 0.068; letter-label 0.314, text-option 0.381), delta vs frozen +0.012 accuracy (57 → 59 of 161), −0.006 unmatched rate, `adapted_beats_frozen` `True` — recorded, not asserted; one seeded split, no dispersion estimate. Section 9: drawn-diagram accuracy frozen 0.4 / adapted 0.4 (same ten answers); adapter 29 tensors, 75,522,608 B, sha256 `e6b175146c25…`; `from_artifact` reload parity 8/8 identical. Preserved output sha256: `pix2struct_ai2d_result.json` `386028cea215…`, `pix2struct_ai2d_evaluation_report.json` `ae50c575687f…`, `pix2struct_ai2d_answers.csv` `16b3d1e7e2fb…`, `pix2struct_ai2d_input_manifest.json` `7e09fb3a3507…`, `pix2struct_ai2d_train.jsonl` `07825a3a0e8d…`, `pix2struct_ai2d_adapter/manifest.json` `920a3ea05129…` | 1173.5 s (pass 1 178.1 s stopped at the install cell with the pip dependency-resolver `CellExecutionError` and the notebook's stale-module guard — `cuda-bindings` 12.9.4 → 13.4.3, `numpy` 2.0.2 → 2.5.3 — kernel restarted after the install cell as step 4 expects; pass 2 995.3 s) | **PASSED** — promotion evidence for this blob |
| 2026-09-26 | `4692451` / `da950ba98d41` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pix2struct-ai2d` v2), same serial suite | Default path | 224.6 s | **FAILED** — 4/11 code cells ok after the install restart; model-staging cell 11 raised `TypeError: _hub_download() takes 1 positional argument but 2 were given` because the standalone notebook's embedded `samples._hub_download` shadowed `pipeline._hub_download`; fixed by the rename in `a633e1a` and the regenerated notebook in `75255e0`. Superseded; not evidence for the current blob |

### Superseded `TASK-INFERENCE` notebook

The rows below are the earlier inference-only notebook's runs; they are history and are **not** evidence for the
`E2E` blob.

### Superseded `TASK-INFERENCE` notebook — local pre-flight (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `fa265aae019c` (commit `9a425ad`, generated at `0bb6466`; `NOTEBOOK_SOURCE.repository_revision` = `0bb6466…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 8 manifest entries (565 MB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (8 files), no font download in the log, ten `answer` calls → `root`, `root`, `root`, `root`, `sun`, `water`, `root`, `sun`, `leaf`, `root` (2 tokens each, none truncated, none unmatched, 2.22–2.34 s), `evaluation_report` `sample-sanity` (`accuracy` 0.4 = 4/10, `unmatched_rate` 0.0, `chance` 0.25; the six recorded misses reproduced), diagram digest `c990244a…` (Pillow 11.3.0 bundled font), 5 outputs written | 85.3 s | PASS — pre-flight only; not promotion evidence |

### Superseded `TASK-INFERENCE` notebook — manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `ba0df32` / `5b2c54315436` | Kaggle CPU (`kurtvalcorza/dimer-nb2-pix2struct-ai2d` v1) | Default sample path | 393.4 s | **PASSED** — 8/8 ok code cells executed cleanly, 18 files, 569 MB staged |

### Hosted Colab T4 runs of blob `be0e3ddf` (2026-10-29)

These runs executed the PR head `930c900de81ef0a50c6503ccbd92243c60d30d72`, notebook `tutorials/pix2struct_ai2d_colab.ipynb` blob `be0e3ddf1e039c55ecf90f59eabdefd0e4f4f2b3`, fetched at the 40-character SHA and Git-blob verified, with the Colab CLI (`colab exec -f`) on a fresh Colab Tesla T4 session (not a browser `Run all`). Each `run_summary.json` records `verdict` `PASS`, `exec_exit_code` 0, `input_unchanged` true and `stop_verified` true (the VM was released), and the SHA-256 of each executed notebook equals the summary's `executed_sha256`.

**Default path (2026-10-29 22:33:28 UTC start).** Default settings, no overrides; session `suite-pix2struct-930c900-30f0`; wall 1049.8 s (`wall_s`); `code_cells` 13, `executed_cells` 10, 0 errors. Retained evidence: [`docs/execution-evidence/2026-10-29-930c900/`](execution-evidence/2026-10-29-930c900/).

Printed by the default run (verbatim lines from the executed outputs):

```text
{'verified_files': 8, 'revision': '0d6b2606efe05c77c1d0670647740802a9e68eef'}
{'frozen_model_test': {'accuracy': 0.354, 'unmatched_rate': 0.075, 'chance': 0.25}, 'n': 161, 'verdict': 'measured', 'adapted': False, 'seconds': 86.0}
{'trainable_parameters': 18879744, 'total_parameters': 282285696, 'training_questions': 360, 'best_epoch': 2, 'selection': 'highest validation accuracy', 'seconds': 670.6}
{'accuracy': {'chance': 0.25, 'position_prior': 0.193, 'longest_option': 0.248, 'frozen': 0.354, 'adapted': 0.366}}
{'delta_vs_frozen': {'accuracy': 0.012, 'unmatched_rate': -0.006}}
{'reload_parity': {'identical_answers': 8, 'of': 8}, 'reloaded_best_epoch': 2}
```

**DAT20 refusal record (2026-10-29, blob `be0e3ddf`).** Overrides, set by `--set` on a copy of the blob: `USE_BYOD` `False` → `True` and `BYOD_PATH` `''` → `'/etc/hostname'` (the run copy differs from the blob in exactly those 2 source lines). The suite cut the run copy right after the refusing cell `pix2struct_ai2d-16`, dropping 19 cells (6 of them code cells), which reproduces `Run all` stopping at the first error; the cut copy is `run_copy.ipynb`, SHA-256 `e2fc151daed308a89863b4668118c18339accf5ffdd31567a23eda124b69637e` (Git blob `fa5ffeeb`), equal to the summary's `run_copy_sha256`. The cut copy's earlier code cells completed without an error output; the refusing cell's only error output has `ename` `IsolatedCellRefusal` and no traceback, and its output holds exactly one `Refused:` line:

```text
Refused: hostname: expected a zip of images plus records.jsonl ending in .zip; fix the input and run this cell again (the earlier results of this session and their outputs/ files were removed).
```

This run says nothing about the 19 cut cells, which did not run in it. Session `suite-pix2struct-930c900-30f0`, wall 111.4 s. Retained evidence: [`docs/execution-evidence/2026-10-29-930c900-refusal/`](execution-evidence/2026-10-29-930c900-refusal/).

**Local-path redaction.** Local paths were replaced byte-wise by `<local-run-dir>` (and the Colab CLI install path by `<colab-cli-tool-dir>/`) before the first commit; each executed notebook and run copy was checked against its `run_summary.json` before any edit.

| File (under `docs/execution-evidence/2026-10-29-930c900/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pix2struct_ai2d_colab_930c900_colab-cli-t4_default.ipynb` | `1b13c5470ff6dfe849fb64cc4c6a219bc62c862f34200b8f9188e39ee283e0a6` | `1b13c5470ff6dfe849fb64cc4c6a219bc62c862f34200b8f9188e39ee283e0a6` (unchanged, no local path) |
| `exec.log` | `09436fdf5a41e741a2f288cc6eee0b6f45d2337b2807895d5a686a9115c4aab4` | `8e83356f14adb5669e607907bb39583588e514a54ed60349953633c5ef093acf` (2 local-path replacements) |
| `run_summary.json` | `58ba7ed39d4dccca0e3e27e61e6e04aa5cb9ef8f4ad517c0b011936dc0238b5d` | `58ba7ed39d4dccca0e3e27e61e6e04aa5cb9ef8f4ad517c0b011936dc0238b5d` (unchanged, no local path) |

| File (under `docs/execution-evidence/2026-10-29-930c900-refusal/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pix2struct_ai2d_colab_930c900_colab-cli-t4_refusal.ipynb` | `9036e5e9b356e1c5aa342a982daaa3dd583b3a86df3b653d762994cf4c28868c` | `9036e5e9b356e1c5aa342a982daaa3dd583b3a86df3b653d762994cf4c28868c` (unchanged, no local path) |
| `exec.log` | `400cd6c4cf89a94b6413b756fc1ee60ffaa464bf4ca09588b03743c83b9f9564` | `0ee6cc63eb1dcfef3bb2b811a627f85f8c155f5276f824c24588722a9e80e2a5` (2 local-path replacements) |
| `run_summary.json` | `58b6fea4e2852e7a0f03e0a532d4e252fe8d975c37c0d29b30d58171c0f46ffb` | `58b6fea4e2852e7a0f03e0a532d4e252fe8d975c37c0d29b30d58171c0f46ffb` (unchanged, no local path) |
| `run_copy.ipynb` | `e2fc151daed308a89863b4668118c18339accf5ffdd31567a23eda124b69637e` | `e2fc151daed308a89863b4668118c18339accf5ffdd31567a23eda124b69637e` (unchanged, no local path) |

Not exercised by these runs: an accepted BYOD input (REL12), a browser `Run all` of the interactive journeys (REL2), and every other open gate named in this file; these runs close none of them.

## Current status

**Candidate.** The current blob `be0e3ddf` (code commit `24ba785`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); its default path (2026-10-29; 13 code cells, exit 0, 0 errors) and its scripted DAT20 refusal (2026-10-29; `USE_BYOD=True;BYOD_PATH=/etc/hostname`: exactly one `Refused:` line at cell `pix2struct_ai2d-16`, no traceback, earlier cells clean, VM released) both passed on a hosted Colab Tesla T4 (Colab CLI `colab exec -f`, not a browser `Run all`), and the hosted runs of earlier blobs, including `bddb3724`, are history for it (REL14). Before that change: The notebook was regenerated at `39d8e03` (blob `bddb37243376`) and that exact blob passed the Colab CLI
Tesla T4 run recorded above (2026-10-08 UTC, 13/13 code cells, 1029.1 s, one pass, no restart, 0 errors, default path
only), reproducing every recorded measurement; promotion to Release-grade is a review decision for the exact release
revision. The earlier Release-grade record identifies commit `75255e0` / blob `02cd3fb8fda4` only and remains history:
it was promotion evidence for that blob, on the passing Kaggle Tesla T4 run recorded
above (2026-09-26 UTC, 11/11 code cells, 1173.5 s). The run exercised every step of the procedure: a clean runtime
with no checkout and an empty cache, the default form parameters, the pinned shard accepted at its recorded SHA-256,
all nine sections, the six exports and the reload-parity assertion. The measured values are one seeded split of 161
held-out AI2D test questions on one runtime, not an AI2D benchmark: frozen accuracy 0.354 and adapted 0.366 against
chance 0.25 and the best non-neural baseline 0.248, so the adaptation gain (+0.012, two questions) is within what one
split without a dispersion estimate can resolve. The first T4 attempt on `4692451` failed on the embedded-module
`_hub_download` name collision, fixed in `a633e1a` / `75255e0`. Any change to the notebook blob returns the carrier to
Candidate until a new exact-blob run is recorded here.

The AI2D shard's SHA-256 pin is recorded in `samples.py` (`tools/pin_corpus.py`; 1,544 rows), and AI2D's licence
terms were reviewed and cleared for this use by the maintainer (Kurt Valcorza, 2026-09-26).

Facts a reviewer should still weigh: the Hub mirror declares no licence for AI2D; its terms were reviewed and cleared
for this use by the maintainer on 2026-09-26, and the notebook still tells users to check them before redistributing
the diagrams or an adapter trained on them; the checkpoint was fine-tuned on AI2D's training questions, so this is continued adaptation inside the
domain and a small or zero gain is the expected outcome, not a defect; the notebook still records
`adapted_beats_frozen` instead of asserting a gain — the recipe (`LEARNING_RATE = 1e-5`, three epochs, two blocks,
batch 4) has now been measured once above, and whether that single measurement is enough to restore an assertion is a
maintainer decision that would change the notebook blob; accuracy counts an unmatched answer as wrong, so teaching the
decoder to copy option text exactly can raise accuracy without better reading of the diagram (the unmatched rate fell
from 0.075 to 0.068 beside the accuracy gain, and letter-label accuracy fell from 0.343 to 0.314); the prompt header
uses Pillow's bundled font rather than the Arial the checkpoint was trained with; each question costs its own encoder
pass at up to 2,048 patches, in training as well as evaluation (adaptation took 688.6 s on the T4); and the
82-question validation and 161-question test splits carry no dispersion estimate.
