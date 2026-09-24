# Release verification

`tutorials/pix2struct_ai2d_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
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
   (= `pyproject.toml`; an interpreter restart after the install is expected where the runtime's preinstalled
   torch or numpy differ from the pins);
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

No execution of the `E2E` notebook is recorded yet. The rows below are the earlier inference-only notebook's runs;
they are history and are **not** evidence for the `E2E` blob.

### Superseded `TASK-INFERENCE` notebook — local pre-flight (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `fa265aae019c` (commit `9a425ad`, generated at `0bb6466`; `NOTEBOOK_SOURCE.repository_revision` = `0bb6466…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 8 manifest entries (565 MB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (8 files), no font download in the log, ten `answer` calls → `root`, `root`, `root`, `root`, `sun`, `water`, `root`, `sun`, `leaf`, `root` (2 tokens each, none truncated, none unmatched, 2.22–2.34 s), `evaluation_report` `sample-sanity` (`accuracy` 0.4 = 4/10, `unmatched_rate` 0.0, `chance` 0.25; the six recorded misses reproduced), diagram digest `c990244a…` (Pillow 11.3.0 bundled font), 5 outputs written | 85.3 s | PASS — pre-flight only; not promotion evidence |

### Superseded `TASK-INFERENCE` notebook — manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `ba0df32` / `5b2c54315436` | Kaggle CPU (`kurtvalcorza/dimer-nb2-pix2struct-ai2d` v1) | Default sample path | 393.4 s | **PASSED** — 8/8 ok code cells executed cleanly, 18 files, 569 MB staged |

## Current status

**Candidate.** No execution of the `E2E` notebook is recorded, and none can be until the AI2D shard's SHA-256 pin is
recorded: `fetch_corpus` refuses to read an unpinned shard, and this repository's build environment had no Hub
access to compute the pin (`tools/pin_corpus.py` records it and prints the realised split counts; regenerate the
notebook afterwards). What exists: static validation (`tools/validate_release_assets.py`), the generator parity checks
(`--check` OK), the offline suites and the adaptation suite on a small random Pix2Struct. The Kaggle CPU and local runs
above were of the earlier `TASK-INFERENCE` notebook, whose inference path (staging, verification, the drawn diagram)
the `E2E` notebook still carries as Section 5, but they do not carry over to the new blob.

Facts a reviewer should weigh before promotion: the Hub mirror declares no licence for AI2D, so AI2D's own terms must
be checked and recorded before the notebook downloads it by default; the checkpoint was fine-tuned on AI2D's training
questions, so this is continued adaptation inside the domain and a small or zero gain is the expected outcome, not a
defect; the fine-tuning recipe (`LEARNING_RATE = 1e-5`, three epochs, two blocks, batch 4) has not been run on this
checkpoint, so the notebook records `adapted_beats_frozen` instead of asserting a gain — restore an assertion once a
measured recipe is recorded here; accuracy counts an unmatched answer as wrong, so teaching the decoder to copy option
text exactly can raise accuracy without better reading of the diagram (the unmatched rate is reported beside it); the
prompt header uses Pillow's bundled font rather than the Arial the checkpoint was trained with, which may cost the
frozen model accuracy that fine-tuning then recovers; each question costs its own encoder pass at up to 2,048
patches, in training as well as evaluation, so a GPU runtime is recommended; and the ~80-question validation and
~160-question test splits carry no dispersion estimate.
