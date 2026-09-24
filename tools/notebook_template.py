"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E diagram multiple-choice workflow: the pinned google/pix2struct-ai2d-base
snapshot is digest-verified and loaded, one digest-pinned parquet shard of AI2D test questions is downloaded,
validated and split by diagram, a drawn diagram with authored questions is answered through the inference
contract, the frozen model is scored on the held-out questions beside chance and two non-neural baselines, a
bounded fine-tuning of the answer decoder's last blocks runs in the kernel, the held-out split is scored again
per category, the adapted model re-answers the drawn diagram, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "pix2struct_ai2d_pipeline",
    "repo_name": "pix2struct-ai2d-pipeline",
    "stem": "pix2struct_ai2d",
    "notebook_name": "pix2struct_ai2d_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `google/pix2struct-ai2d-base` snapshot (a 565 MB `model.safetensors`), downloads one digest-pinned parquet "
        "shard of AI2D test questions from the Hugging Face Hub (62 MB, no credential, refused on any size or SHA-256 "
        "mismatch), cuts a seeded subset of whole diagrams into training, validation and test questions so no diagram is "
        "shared, answers ten authored questions on a drawn plant diagram through the inference contract with an input "
        "manifest and a rejection probe, scores the frozen model on the test questions beside chance and two non-neural "
        "baselines, runs a bounded fine-tuning of the answer decoder's last blocks with validation-accuracy epoch selection, "
        "scores the held-out questions again per category, re-answers the drawn diagram with the adapted model, exports the "
        "adapter as safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify answer parity. The "
        "default path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no "
        "configuration edit (NOTEBOOK_SPEC 2.0 §5). A CUDA runtime is used automatically when present; the CPU path works but "
        "is slow (every question renders its own header and is encoded at up to 2,048 patches), and the timings of the first "
        "clean run are recorded in `docs/release-verification.md`."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "holding a `records.jsonl` (or `records.json`) of `{{id, image, question, options, answer}}` objects — `image` a file "
        "name inside the zip, `options` 2–6 distinct strings, `answer` the zero-based index of the correct option, optional "
        "`image_id` and `category` — beside the image files. They pass through the same validation, seeded diagram-disjoint "
        "split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the AI2D sample. The "
        "expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside "
        "this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "Pix2StructAI2DPipeline",
    "weights_key": "pix2struct-ai2d-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "Pix2Struct AI2D-base — DIMER E2E diagram multiple-choice fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/pix2struct-ai2d-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/pix2struct-ai2d-pipeline/blob/main/tutorials/pix2struct_ai2d_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fpix2struct--ai2d--base-ffcc4d?style=flat",
            "https://huggingface.co/google/pix2struct-ai2d-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Fpix2struct-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/pix2struct",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2210.03347-b31b1b.svg", "https://arxiv.org/abs/2210.03347"),
    ],
    "capability": "diagram multiple-choice question answering and bounded supervised fine-tuning of the answer decoder's last blocks on a diagram/question/options dataset, using the pinned `google/pix2struct-ai2d-base` weights",
    "intro": (
        "`google/pix2struct-ai2d-base` is the Pix2Struct model of Lee et al. (2023) — a ViT-style image encoder over "
        "variable-resolution 16×16 patches (up to 2,048 per image) and a 12-layer text decoder that cross-attends to them; "
        "282,285,696 parameters, pretrained by parsing masked web screenshots into simplified HTML and fine-tuned on AI2D, "
        "a set of school-science diagrams with multiple-choice questions — published under the **Apache-2.0** licence. The "
        "question and its numbered options are **rendered as a text header above the diagram** (the Pix2Struct convention "
        "for visual question answering, `<question> (1) <a> (2) <b> …`, in Pillow's bundled font so no font is downloaded), "
        "the composite is encoded, and the decoder generates the answer text with greedy decoding under a caller-owned "
        "`max_new_tokens` budget; the carried module matches that text to the options by normalised exact match. **No score "
        "exists**: the answer is generated text with no probability and no correctness signal, and matching an option is "
        "**not evidence that it is the right one**.\n\n"
        "What this notebook adds to inference is **adaptation with labelled questions**. The dataset is real and from the "
        "checkpoint's own domain: AI2D test questions (Kembhavi et al., ECCV 2016) as mirrored on the Hub — the checkpoint "
        "was fine-tuned on AI2D's *training* questions, so these diagrams and questions are unseen but not out of "
        "distribution, and the honest question is a narrow one: does a bounded adaptation of the answer decoder's last "
        "blocks on a few hundred more in-domain questions move held-out accuracy at all, and on which kind of question? "
        "About a third of AI2D questions ask about lettered diagram labels (`A`, `B`, `C`, `D` as options), the rest have "
        "text options, so the per-category breakdown (`letter-label` / `text-option`) is part of the reading. The notebook "
        "downloads **one pinned parquet shard** (62 MB, SHA-256 pinned in the carried module) and draws a seeded subset of "
        "whole diagrams from it. Accuracy counts an answer that matches no option as wrong, and three references frame it: "
        "**chance** (the mean of 1/options), the **position-prior** baseline (always the option position most often correct "
        "in training) and the **longest-option** baseline — two systems that never look at the diagram. Nothing here is a "
        "quality claim about your diagrams: it is one seeded split of one shard.\n\n"
        "**Weight-format note:** the pinned revision ships the model as SafeTensors (`model.safetensors`, stored in bfloat16 "
        "and upcast to float32 at load, digest-pinned in the manifest); the processor is the VQA variant, which renders the "
        "header. Section 3 stages and digest-verifies the snapshot before the processor or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; download a digest-pinned shard of labelled diagram questions, "
        "validate it and split it by diagram without leakage; answer through the public API on a drawn diagram with authored "
        "questions and read `answer`, `choice_index`, `new_tokens` and `truncated` correctly (generated text matched to an "
        "option or to none, no score); score the frozen model's accuracy beside chance and two non-neural baselines and read "
        "the `letter-label` / `text-option` breakdown; run a bounded fine-tuning with explicit hyperparameters and "
        "validation-based epoch selection; evaluate on a diagram-disjoint test split; re-answer a drawing from a different "
        "image family with the adapted model; and export a safetensors adapter that reloads against the pinned base with "
        "verified parity."
    ),
    "exclusions": (
        "Free-form or open-ended answers (the contract matches the generated text to the supplied options and reports `null` "
        "otherwise), document, chart or scene question answering (separate checkpoints), reading a diagram's text back as a "
        "transcript, answer localisation, batch throughput, sampling or beam search (the notebook decodes greedily for "
        "reproducibility), evaluation on the full AI2D benchmark (only a seeded subset of one shard is scored here), "
        "fine-tuning of the image encoder, the embeddings or the output projection, training on diagrams that are not the "
        "pinned sample or your own uploads, and any claim that an AI2D split stands in for your diagrams. The repository "
        "exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available; a GPU runtime is recommended for Sections 6–8. Every question renders its own header above its diagram and is encoded at up to 2,048 patches, so each answer costs seconds on CPU. The pinned `torch==2.14.0` install and the 565 MB checkpoint are the large downloads of the run; the question shard adds 62 MB.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; what accuracy against chance and against baselines that never see the image does and does not show; why a confident answer is not a correct one.",
        "- **Data contract:** records are `{{id, image, question, options, answer}}` — an image file decodable by Pillow with sides between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` (4096) px, a non-empty question of at most `MAX_QUESTION_CHARS` (256) characters, `MIN_OPTIONS`..`MAX_OPTIONS` (2..6) distinct options of at most `MAX_OPTION_CHARS` (64) characters, and `answer` the zero-based index of the correct option; optional `image_id` groups questions on the same diagram (BYOD defaults it to the image file name) and optional `category` labels the breakdown. Ids match `[A-Za-z0-9_.:-]{{1,64}}` and are unique; a dataset needs 8..5,000 records; every question on the same diagram lands in the same split so a test diagram is never trained on. BYOD accepts one zip of images plus a `records.jsonl` / `records.json` in that shape.",
        "- **Validation is structural, not semantic:** every diagram is opened and decoded and every question checked against the same ceilings `answer` applies, but nothing checks that the marked answer is right — a mislabelled question is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path downloads one object from the Hub dataset repository `lmms-lab-encoder/ai2d` at the immutable revision `c83a9b96…` (`data/test-00000-of-00002.parquet`, 62,292,686 bytes) and refuses it unless its size and SHA-256 match the pins carried in `samples.py`; the diagrams are written to the cache under their own content digest. The Hub mirror declares no licence; AI2D is published by the Allen Institute for AI (Kembhavi et al., 2016) — check its terms before redistributing the diagrams or an adapter trained on them.",
    ],
    "cells": [
        {
            "md": (
                "## 4. AI2D questions, diagrams and split\n\n"
                "`fetch_corpus` returns the pinned shard from the cache under `weights/ai2d/` or downloads it at the pinned "
                "dataset revision, and refuses it unless its byte size and SHA-256 equal the pins in the carried module (it "
                "also refuses to run at all while no SHA-256 pin is recorded). `read_corpus` reads the question, options, "
                "answer and image columns with `pyarrow`. `build_sample_dataset` groups the questions by diagram (the "
                "SHA-256 of the image bytes), shuffles the diagrams with `SPLIT_SEED` and allocates **whole diagrams** to the "
                "test, validation and training splits until each reaches its question target (`SAMPLE_QUESTIONS`), leaving out "
                "the few questions `validate_dataset` would reject (an option longer than `MAX_OPTION_CHARS`, or options that "
                "coincide after normalisation), labelling "
                "each question `letter-label` when every option is a one- or two-character diagram label and `text-option` "
                "otherwise. `validate_dataset` then opens and decodes every diagram and checks every question against the "
                "contract, `check_split_disjoint` asserts no diagram is shared, and the training split is written to "
                "`outputs/{stem}_train.jsonl` in the shape BYOD expects.\n\n"
                "Look for: the shard's row count, the question and diagram counts per split, the category mix, the "
                "distribution of correct positions (the position-prior baseline in Section 6 is built from it), three "
                "digests, and four refusal probes — a duplicate id, a missing image file, an answer index out of range and a "
                "dataset too small to split — each rejected before `torch` does anything."
            ),
            "code": (
                "import collections\n"
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import zipfile\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_dir = Path('work') / 'byod'\n"
                "    byod_dir.mkdir(parents=True, exist_ok=True)\n"
                "    with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
                "        for member in archive.infolist():\n"
                "            name = Path(member.filename).name\n"
                "            if member.is_dir() or not name or name.startswith('.'):\n"
                "                continue\n"
                "            (byod_dir / name).write_bytes(archive.read(member))\n"
                "    records_file = next(p for p in (byod_dir / 'records.jsonl', byod_dir / 'records.json') if p.is_file())\n"
                "    records = load_byod_dataset(records_file)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED, base_dir=byod_dir)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    shard_path = fetch_corpus(cache_dir='weights/ai2d')\n"
                "    corpus_rows = read_corpus(shard_path)\n"
                "    raw_rows = {{'questions': len(corpus_rows), 'diagrams': len({{hashlib.sha256(r['image_bytes']).hexdigest() for r in corpus_rows}})}}\n"
                "    splits = build_sample_dataset(corpus_rows, seed=SPLIT_SEED, image_dir='weights/ai2d/images')\n"
                "    data_source = f'{{CORPUS_NAME}} — {{CORPUS_RELEASE}}'\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "categories = {{name: manifest['categories'] for name, manifest in dataset_manifests.items()}}\n"
                "write_dataset_jsonl(splits['train'], 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'shard_sha256': str(CORPUS_FILE['sha256'])[:16] + '...'}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'questions': manifest['n_records'], 'diagrams': manifest['unique_images'], 'categories': manifest['categories'], 'answer_positions': manifest['answer_positions'], 'options_per_question': manifest['options_per_question'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'image': Path(example['image']).name, 'size': example['image_size'], 'category': example['category'], 'question': example['question'], 'options': example['options'], 'answer': example['answer']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'missing image file': [{{**splits['train'][0], 'image': 'work/does-not-exist.png'}}, *splits['train'][1:8]],\n"
                "    'answer out of range': [{{**splits['train'][0], 'answer': len(splits['train'][0]['options'])}}, *splits['train'][1:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Answer through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a flat cartoon plant diagram "
                "drawn in code at 640×640 — a pink flower, two leaves, a stem, four roots in brown soil and a sun — with six "
                "numbered markers joined to their parts (the AI2D convention: numbers on the diagram, words only in the "
                "options), and ten authored questions with their correct options. It is a different image family from the "
                "AI2D diagrams, and the adapted model will be asked the same questions in Section 9. `validate_inputs` applies "
                "exactly the checks `answer` applies (image sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, a non-empty question of at "
                "most `MAX_QUESTION_CHARS`, `MIN_OPTIONS`..`MAX_OPTIONS` distinct options of at most `MAX_OPTION_CHARS`, "
                "`max_new_tokens` in `[1, MAX_NEW_TOKENS]`) and returns an input manifest; a question with a single option is "
                "validated too and its rejection recorded as a finding. `answer` returns the decoded text, the matched "
                "`choice_index` (or `null` when the text equals no option — there is no fuzzy match), the rendered prompt, "
                "`new_tokens`, a `truncated` flag and the model identity. **No score exists.** As recorded in the model card, "
                "the inference-only smoke answered four of the ten correctly (chance is one in four) and said `root` to almost "
                "every label question whenever `root` was an option. `evaluation_report` scores those ten against the options "
                "you drew yourself — verdict `sample-sanity`, plumbing evidence, not a metric; whether answers are *right* on "
                "real diagrams is what Section 6 measures on the test questions. The image digest depends on the Pillow build's "
                "bundled font rendering."
            ),
            "code": (
                "import math\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "ANSWER_MAX_TOKENS = 16  # @param {{type:\"integer\"}}\n\n\n"
                "def synthetic_diagram(size=640):\n"
                "    \"\"\"A cartoon plant diagram with six numbered markers; returns image + [(question, options, correct index)].\"\"\"\n"
                "    image = Image.new('RGB', (size, size), 'white')\n"
                "    d = ImageDraw.Draw(image)\n"
                "    marker_font = ImageFont.load_default(size=34)\n"
                "    d.rectangle([0, 440, 640, 640], fill=(160, 120, 70))  # soil\n"
                "    d.ellipse([500, 40, 600, 140], fill=(255, 215, 0))  # sun\n"
                "    d.rectangle([310, 220, 330, 440], fill=(40, 140, 40))  # stem\n"
                "    d.polygon([(310, 330), (220, 290), (240, 350)], fill=(50, 170, 50))  # left leaf\n"
                "    d.polygon([(330, 380), (420, 340), (400, 400)], fill=(50, 170, 50))  # right leaf\n"
                "    for k in range(6):  # petals\n"
                "        a = math.radians(60 * k)\n"
                "        cx, cy = 320 + 45 * math.cos(a), 190 + 45 * math.sin(a)\n"
                "        d.ellipse([cx - 22, cy - 22, cx + 22, cy + 22], fill=(230, 60, 120))\n"
                "    d.ellipse([298, 168, 342, 212], fill=(255, 200, 40))  # flower centre\n"
                "    for dx in (-60, -20, 25, 70):  # roots\n"
                "        d.line([(320, 440), (320 + dx, 560)], fill=(120, 80, 40), width=5)\n"
                "    markers = [('1', (140, 150), (275, 175)), ('2', (90, 330), (225, 320)), ('3', (470, 250), (332, 300)), ('4', (140, 560), (290, 520)), ('5', (550, 175), (550, 140)), ('6', (560, 600), (560, 600))]\n"
                "    for text, pos, tip in markers:\n"
                "        if pos != tip:\n"
                "            d.line([pos, tip], fill='black', width=3)\n"
                "        d.rectangle([pos[0] - 22, pos[1] - 22, pos[0] + 22, pos[1] + 22], fill='white', outline='black', width=2)\n"
                "        d.text(pos, text, fill='black', font=marker_font, anchor='mm')\n"
                "    qa = [\n"
                "        ('What does the label 1 represent?', ['flower', 'leaf', 'stem', 'root'], 0),\n"
                "        ('What does the label 2 represent?', ['flower', 'leaf', 'stem', 'root'], 1),\n"
                "        ('What does the label 3 represent?', ['root', 'stem', 'leaf', 'flower'], 1),\n"
                "        ('What does the label 4 represent?', ['stem', 'flower', 'root', 'leaf'], 2),\n"
                "        ('What does the label 5 represent?', ['moon', 'sun', 'cloud', 'rain'], 1),\n"
                "        ('What does the label 6 represent?', ['water', 'air', 'soil', 'rock'], 2),\n"
                "        ('Which part of the plant is below the soil?', ['flower', 'leaf', 'stem', 'root'], 3),\n"
                "        ('What provides light to the plant?', ['soil', 'sun', 'root', 'leaf'], 1),\n"
                "        ('Which part connects the roots to the flower?', ['leaf', 'soil', 'stem', 'sun'], 2),\n"
                "        ('Which part is at the top of the plant?', ['root', 'stem', 'flower', 'soil'], 2),\n"
                "    ]\n"
                "    return image, qa\n\n\n"
                "diagram, qa = synthetic_diagram()\n"
                "diagram_name = 'synthetic_plant_diagram_640x640.png'\n"
                "diagram_questions = [{{'question': q, 'options': options}} for q, options, _ in qa]\n"
                "diagram_correct = [gold for _, _, gold in qa]\n"
                "diagram_sha256 = hashlib.sha256(np.asarray(diagram.convert('RGB')).tobytes()).hexdigest()\n"
                "ceilings = {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MIN_OPTIONS': MIN_OPTIONS, 'MAX_OPTIONS': MAX_OPTIONS, 'MAX_OPTION_CHARS': MAX_OPTION_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(diagram, diagram_questions, max_new_tokens=ANSWER_MAX_TOKENS, names=[diagram_name])\n"
                "try:\n"
                "    validate_inputs(diagram, [{{'question': 'What is this?', 'options': ['a plant']}}])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'single-option-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'diagram': diagram_name, 'rgb_sha256': diagram_sha256[:16] + '...', 'questions': len(diagram_questions), 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "results = []\n"
                "for entry in diagram_questions:\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.answer(diagram, entry['question'], entry['options'], max_new_tokens=ANSWER_MAX_TOKENS)\n"
                "    results.append({{'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "for result in results:\n"
                "    matched = result['options'][result['choice_index']] if result['choice_index'] is not None else 'NO OPTION MATCHED'\n"
                "    print(f\"Q: {{result['question']}}  options={{result['options']}}\\n   A: {{result['answer']!r}} -> {{matched}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\")\n"
                "checks = {{\n"
                "    'one_result_per_question': len(results) == len(diagram_questions),\n"
                "    'answers_are_text': all(isinstance(r['answer'], str) for r in results),\n"
                "    'budget_respected': all(r['new_tokens'] <= ANSWER_MAX_TOKENS for r in results),\n"
                "    'setting_echoed': all(r['generation']['max_new_tokens'] == ANSWER_MAX_TOKENS and r['generation']['do_sample'] is False for r in results),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'answer output failed a sanity check: {{checks}}')\n"
                "frozen_diagram = evaluation_report(results, diagram_correct, sample_kind='synthetic')\n"
                "print({{'checks': checks, 'frozen_diagram_verdict': frozen_diagram['verdict'], 'accuracy': frozen_diagram['metrics'][0]['value'], 'chance': frozen_diagram['baselines'][0]['value'], 'unmatched': sum(r['choice_index'] is None for r in results)}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's accuracy on the test questions\n\n"
                "Four references frame the adaptation. **Chance** is the mean of 1/options over the test questions — what "
                "uniform guessing scores. The **position-prior baseline** answers every test question with the option "
                "position most often correct in the training split (printed in Section 4). The **longest-option baseline** "
                "answers with the longest option text. Neither baseline looks at the diagram, so a model that does not beat "
                "them has not shown that it reads the diagram. The **frozen model** answers every test question with the "
                "budget from Section 5 and is scored by `pipe.evaluate`: **accuracy** (the matched option is the correct one; "
                "an answer that matches no option counts as wrong), the **unmatched rate** and a per-category accuracy "
                "(`letter-label` questions name lettered parts of the diagram; `text-option` questions have words as options). "
                "The checkpoint was fine-tuned on AI2D's training questions, so expect it well above chance here; whether it "
                "is, is recorded as `frozen_beats_chance` rather than assumed. The measured values of the first clean run are "
                "recorded in `docs/release-verification.md` and the model card."
            ),
            "code": (
                "baseline_position = position_prior_baseline(train_records, test_records)\n"
                "baseline_longest = longest_option_baseline(test_records)\n"
                "print({{'position_prior_baseline': round(baseline_position['accuracy'], 3), 'note': baseline_position['note'], 'n': baseline_position['n']}})\n"
                "print({{'longest_option_baseline': round(baseline_longest['accuracy'], 3), 'chance': round(baseline_longest['chance'], 3)}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=ANSWER_MAX_TOKENS)\n"
                "print({{'frozen_model_test': {{'accuracy': round(frozen_test['accuracy'], 3), 'unmatched_rate': round(frozen_test['unmatched_rate'], 3), 'chance': round(frozen_test['chance'], 3)}}, 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'by_category': {{'position_prior': baseline_position['by_category'], 'frozen': frozen_test['by_category']}}}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "frozen_predictions = {{r['id']: r for r in pipe.predict(test_records[:3], max_new_tokens=ANSWER_MAX_TOKENS)}}\n"
                "for record in test_records[:3]:\n"
                "    print({{'category': record['category'], 'question': record['question'], 'options': record['options'], 'correct': record['options'][record['answer']], 'frozen': frozen_predictions[record['id']]['answer']}})\n"
                "frozen_beats_chance = frozen_test['accuracy'] > frozen_test['chance']\n"
                "print({{'frozen_beats_chance': frozen_beats_chance}})"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the answer decoder's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_DECODER_LAYERS` blocks of the answer decoder plus the decoder's "
                "final layer norm — two blocks by default, 18,879,744 of 282,285,696 parameters; the image encoder, every "
                "embedding and the untied output projection (a 50,244 × 768 matrix) stay frozen. Each training question is one "
                "sample: its question and numbered options are rendered above its diagram exactly as `answer` renders them, "
                "the frozen encoder reads the composite (recomputed each step without gradients — the header makes every "
                "question its own image), and the target is the tokenised text of the correct option with its end-of-sequence "
                "token, decoded with teacher forcing and scored with the model's own cross-entropy (padding ignored); AdamW at "
                "a fixed learning rate, gradient clipping at 1.0, seeded shuffling and no scheduler. Epoch 0 records the frozen "
                "model's validation accuracy; every epoch is scored on the validation questions and the epoch with the highest "
                "validation accuracy is kept (ties keep the earlier one). A validation split of about eighty questions makes "
                "that selection coarse — one question is more than a point of accuracy — which is why the held-out split in "
                "Section 8 is what the numbers are read from. If no epoch beats the frozen model on validation, the selector "
                "keeps epoch 0 and the adapter reproduces the frozen answers; that outcome is reported, not hidden."
            ),
            "code": (
                "EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_accuracy': round(entry['val']['accuracy'], 3), 'val_unmatched_rate': round(entry['val']['unmatched_rate'], 3)}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'training_questions': adapt_result['n_train'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test questions were never used for training or epoch selection, and no test diagram appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6, and "
                "the five systems — chance, the two baselines, the frozen and the adapted model — are put side by side overall "
                "and per category. Read it in this order: **accuracy** first (the metric the epoch was selected on), then the "
                "**unmatched rate** (an adaptation that teaches the decoder to copy option text exactly lowers it, which raises "
                "accuracy without the model reading the diagram any better), then the `letter-label` / `text-option` split. "
                "`adapted_beats_frozen` records whether held-out accuracy rose. A test split of about 160 questions from one "
                "seeded draw of one shard gives **no dispersion estimate** — one question is more than half a point — so the "
                "deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and a gain on AI2D "
                "says nothing about your diagrams until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=ANSWER_MAX_TOKENS)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=ANSWER_MAX_TOKENS)\n"
                "comparison = {{\n"
                "    'accuracy': {{'chance': round(frozen_test['chance'], 3), 'position_prior': round(baseline_position['accuracy'], 3), 'longest_option': round(baseline_longest['accuracy'], 3), 'frozen': round(frozen_test['accuracy'], 3), 'adapted': round(adapted_test['accuracy'], 3)}},\n"
                "    'unmatched_rate': {{'frozen': round(frozen_test['unmatched_rate'], 3), 'adapted': round(adapted_test['unmatched_rate'], 3)}},\n"
                "    'delta_vs_frozen': {{'accuracy': round(adapted_test['accuracy'] - frozen_test['accuracy'], 3), 'unmatched_rate': round(adapted_test['unmatched_rate'] - frozen_test['unmatched_rate'], 3)}},\n"
                "    'by_category': {{category: {{'n': row['n'], 'position_prior': round(baseline_position['by_category'][category]['accuracy'], 3), 'frozen': round(row['accuracy'], 3), 'adapted': round(adapted_test['by_category'][category]['accuracy'], 3)}} for category, row in frozen_test['by_category'].items()}},\n"
                "}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "adapted_predictions = {{r['id']: r for r in pipe.predict(test_records[:3], max_new_tokens=ANSWER_MAX_TOKENS)}}\n"
                "for record in test_records[:3]:\n"
                "    print({{'question': record['question'], 'correct': record['options'][record['answer']], 'frozen': frozen_predictions[record['id']]['answer'], 'adapted': adapted_predictions[record['id']]['answer']}})\n"
                "adapted_beats_frozen = adapted_test['accuracy'] > frozen_test['accuracy']\n"
                "print({{'adapted_beats_frozen': adapted_beats_frozen}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'categories': categories,\n"
                "    'max_new_tokens': ANSWER_MAX_TOKENS,\n"
                "    'baselines': {{'position_prior': baseline_position, 'longest_option': baseline_longest}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'frozen_beats_chance': frozen_beats_chance,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "    'adapted_beats_frozen': adapted_beats_frozen,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-answer the drawn diagram, export the adapter and reload it\n\n"
                "The ten questions on the drawn plant from Section 5 are answered again by the adapted model and scored against "
                "the options you drew — a different image family from the AI2D diagrams it was tuned on, so this is a small look "
                "at whether the adaptation changed the model's behaviour *outside* its sample (ten questions of evidence, not a "
                "measurement; a different answer here is a finding to record, not a failure). Both answer sets are written as "
                "CSV.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the answer decoder's last two blocks and its final layer norm, "
                "about 76 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id "
                "and revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `Pix2StructAI2DPipeline.from_artifact` re-verifies the base "
                "snapshot, checks the artifact manifest, its digest and its exact tensor set **before** deserialising, refuses "
                "any tensor that is not an answer-decoder tensor, and overlays the tensors onto a freshly loaded base — a new "
                "object from files, not the in-memory model (VER2). The cell asserts identical answers on eight test questions "
                "(VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "adapted_results = [pipe.answer(diagram, entry['question'], entry['options'], max_new_tokens=ANSWER_MAX_TOKENS) for entry in diagram_questions]\n"
                "adapted_diagram = evaluation_report(adapted_results, diagram_correct, sample_kind='synthetic')\n"
                "for before, after in zip(results, adapted_results, strict=True):\n"
                "    print({{'question': before['question'], 'frozen': before['answer'], 'adapted': after['answer']}})\n"
                "print({{'drawn_diagram_accuracy': {{'frozen': frozen_diagram['metrics'][0]['value'], 'adapted': adapted_diagram['metrics'][0]['value']}}}})\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'question', 'options', 'correct_option', 'frozen_answer', 'adapted_answer', 'adapted_choice_index'])\n"
                "    for before, after, gold in zip(results, adapted_results, diagram_correct, strict=True):\n"
                "        writer.writerow([diagram_name, before['question'], ' | '.join(before['options']), before['options'][gold], before['answer'], after['answer'], after['choice_index']])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = Pix2StructAI2DPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [r['answer'] for r in pipe.predict(test_records[:8], max_new_tokens=ANSWER_MAX_TOKENS)]\n"
                "after = [r['answer'] for r in reloaded.predict(test_records[:8], max_new_tokens=ANSWER_MAX_TOKENS)]\n"
                "parity = {{'identical_answers': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_answers'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'SafeTensors (bfloat16, upcast to float32), digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'file': CORPUS_FILE, 'sample_questions': SAMPLE_QUESTIONS}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'diagram': {{'name': diagram_name, 'size': list(diagram.size), 'rgb_sha256': diagram_sha256, 'questions': diagram_questions, 'correct': diagram_correct}}, 'items': [{{k: r[k] for k in ('question', 'answer', 'choice_index', 'new_tokens', 'truncated', 'seconds')}} for r in results], 'frozen_report': frozen_diagram, 'adapted_items': [{{k: r[k] for k in ('question', 'answer', 'choice_index', 'new_tokens', 'truncated')}} for r in adapted_results], 'adapted_report': adapted_diagram}},\n"
                "    'comparison': comparison,\n"
                "    'frozen_beats_chance': frozen_beats_chance,\n"
                "    'adapted_beats_frozen': adapted_beats_frozen,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is an AI2D-trained answerer scored on AI2D questions it was not trained on, beside chance and two "
        "baselines that never look at the diagram, and a bounded fine-tuning of the answer decoder's last two blocks on a few "
        "hundred more questions is then scored on a diagram-disjoint test split — overall, and separately on lettered-label "
        "and text-option questions — with an adapter that reloads to identical answers. That is the claim: the adaptation "
        "contract works end to end on a real labelled diagram corpus, and the numbers it produces are read against chance, "
        "the baselines and the frozen model rather than in isolation. Whether held-out accuracy rose is recorded as "
        "`adapted_beats_frozen`, not assumed.\n\n"
        "The test split is about 160 questions on whole diagrams from one seeded draw of one shard, the validation split that "
        "picks the epoch is about 80, and accuracy counts an unmatched answer as wrong — so a lower unmatched rate alone can "
        "raise it. Because the checkpoint already saw AI2D's training questions, a small or zero gain is the expected outcome "
        "and not a failure of the contract; a learning rate that is too high overfits this little data within an epoch, "
        "which the validation-based selector reports by keeping epoch 0. So a gain here says the contract works, not that the "
        "adapted model reads diagrams better, and it still answers every question — including unanswerable ones — with a "
        "fluent option. Fine-tuning on a narrow sample can also erode the model elsewhere; the drawn diagram re-answered in "
        "Section 9 is ten questions of evidence about that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** chance, the position prior and the longest option on *your* "
        "questions are the numbers to read before any model's, per category. **Leakage:** keep every question on a diagram in "
        "one split (the contract does this) and split by source textbook or chapter when your diagrams come from few sources. "
        "**Options are part of the input:** the answer depends on the option set (the inference-only smoke answered `leaf` once "
        "`root` was removed), so a changed option order or wording is a changed request.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled diagram "
        "corpus, validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded "
        "fine-tuning, evaluate against chance, two trivial baselines and the frozen model on a diagram-disjoint split, and emit "
        "the shown machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark "
        "superiority, accuracy on any other diagram population, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** raise `LEARNING_RATE` and watch the training loss "
        "fall while the validation accuracy drops and the selector keeps an early epoch; set `TRAINABLE_DECODER_LAYERS = 1` and "
        "compare the artifact size and the held-out accuracy; shuffle the options of a test question and watch the answer "
        "follow them; or bring your own diagrams through BYOD and read the baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/pix2struct-ai2d-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/pix2struct-ai2d-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/pix2struct-ai2d-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (Google, Apache-2.0): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/pix2struct\n"
        "- Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding (Lee et al., ICML 2023): https://arxiv.org/abs/2210.03347\n"
        "- A Diagram Is Worth A Dozen Images — the AI2D dataset (Kembhavi et al., ECCV 2016): https://arxiv.org/abs/1603.07396\n"
        "- AI2D test questions as mirrored on the Hugging Face Hub: https://huggingface.co/datasets/lmms-lab-encoder/ai2d\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
