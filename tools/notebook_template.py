"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "pix2struct_ai2d_pipeline",
    "repo_name": "pix2struct-ai2d-pipeline",
    "stem": "pix2struct_ai2d",
    "notebook_name": "pix2struct_ai2d_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "Pix2StructAI2DPipeline",
    "weights_key": "pix2struct-ai2d-base",
    "runtime_imports": ["torch", "transformers"],
    "title": "Pix2Struct AI2D-base — DIMER diagram multiple-choice question answering tutorial (standalone)",
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
    "capability": "Diagram multiple-choice question answering — one diagram image plus one question with 2–6 answer options → the generated option text and its matched index — using the pinned `google/pix2struct-ai2d-base` weights",
    "intro": (
        "At inference the Pix2Struct image-encoder/text-decoder (a ViT-style encoder over variable-resolution 16×16 patches "
        "and a 12-layer text decoder, 282M parameters, pretrained by parsing masked web screenshots into HTML and fine-tuned "
        "on AI2D, a set of school-science diagrams with multiple-choice questions) reads the **question and its numbered "
        "options rendered as a text header above the diagram** — the Pix2Struct convention for visual question answering, "
        "with the pinned README's `<question> (1) <a> (2) <b> …` prompt format — scales the composite to fill at most 2048 "
        "patches, and generates the answer text token by token; the carried module then matches that text to the options by "
        "normalised exact match. Decoding is greedy (`do_sample=False`) under a caller-owned `max_new_tokens` budget. **No "
        "adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in this "
        "notebook — the upstream checkpoint supplies the weights, processor and tokenizer, and the carried module adds "
        "snapshot verification, the input contract (image side ceilings, a non-empty question up to 256 characters, 2–6 "
        "distinct options up to 64 characters each, the token budget), a fixed output contract, an offline header font "
        "(Pillow's bundled Aileron replaces the Hub font the upstream processor would otherwise download), and the "
        "`format_prompt`, `match_option`, `validate_inputs` and `evaluation_report` helpers. The default sample is a flat "
        "cartoon plant diagram drawn in code with six numbered markers and ten authored questions, so `accuracy` against the "
        "chance baseline is demonstration (plumbing) evidence for one drawing, not an AI2D benchmark — and the model gets six "
        "of the ten wrong, which the notebook keeps and explains."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic labelled diagram with authored multiple-choice questions (or "
        "upload your own diagram and write your own questions) and validate it into an input manifest, choose a token budget, "
        "run the supported task, read the answers correctly (generated text matched to an option or to none, no score, a "
        "`truncated` flag), exercise an optional BYOD path, produce an evaluation report that is `sample-sanity` with "
        "`accuracy`, `unmatched_rate` and a chance baseline only when the correct options are known and `not-measurable` "
        "otherwise, and export the answers, the annotated diagram and provenance."
    ),
    "exclusions": (
        "Free-form or open-ended answers (the model generates text, but the contract matches it to the supplied options and "
        "reports `null` otherwise), document or scene question answering (separate checkpoints), reading a diagram's text "
        "back as a transcript, answer localisation, batch throughput, sampling or beam search, evaluation on the AI2D "
        "benchmark (not bundled; only authored questions on a drawn diagram are scored here), and any training. The model "
        "was fine-tuned on textbook-style science diagrams with numbered label markers; flat cartoons, photographs, charts "
        "and non-English questions are outside what this notebook measures, and a fluent wrong answer carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both (the checkpoint is stored in bfloat16 and upcast at load). CPU is adequate: the repository's model card records 5.2 s to load and 2.1–2.4 s per question on the 640×640 drawn diagram in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 565 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; what accuracy against a chance baseline does and does not show on ten questions; that a confident answer is not a correct one.",
        "- **Data:** the default sample is a deterministic 640×640 cartoon plant diagram drawn in code (flower, two leaves, stem, roots, soil, sun) with six numbered markers in Pillow's bundled font and ten authored multiple-choice questions with their correct options, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar) of a **single diagram**, any colour mode, sides between 16 and 4096 px, plus your own questions typed into the form field as `question | option | option | …` lines. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic diagram or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: a cartoon plant — a pink flower, two "
                "leaves, a stem, four roots in brown soil and a sun — is drawn with Pillow at 640×640 with six numbered "
                "markers in white boxes, each joined to its part by a line (the AI2D convention: numbers on the diagram, "
                "words only in the options), the same drawing the repository's smoke run used. Ten questions are authored "
                "against it (six `What does the label N represent?` questions and four about the plant), each with its "
                "correct option; they are the references for the `accuracy` sanity check later. They are not a labelled "
                "dataset, so nothing here is an AI2D measurement — and the smoke run answered only four of them correctly "
                "(chance is one in four): the model answers `root` to almost any label question about this cartoon, which the "
                "notebook keeps as a recorded finding rather than tuning the drawing until it passes. The image digest is "
                "printed for the record; it depends on the Pillow build's bundled font rendering. BYOD is optional and "
                "disabled by default; when enabled, upload one diagram and type your questions — no correct options are "
                "known for them, so the evaluation report will be `not-measurable`.\n\n"
                "The token budget is a **caller-owned request parameter**: `max_new_tokens` bounds the answer "
                "(`DEFAULT_MAX_NEW_TOKENS = 16` fits any option; `MAX_NEW_TOKENS = 64` is the ceiling). Nothing is validated "
                "in this cell — the next section hands the image and the questions to the pipeline's own validation stage, "
                "which is the only checker. Look for a dictionary naming the sample kind, the image size and digest, the "
                "budget and the number of questions."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import math\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "byod_questions = 'What does the label 1 represent? | flower | leaf | stem | root'  # @param {{type:\"string\"}}\n"
                "max_new_tokens = 16  # @param {{type:\"integer\"}}\n\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    questions = []\n"
                "    for line in byod_questions.splitlines():\n"
                "        parts = [part.strip() for part in line.split('|') if part.strip()]\n"
                "        if len(parts) >= 3:\n"
                "            questions.append({{'question': parts[0], 'options': parts[1:]}})\n"
                "    correct = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawing: no randomness, so no seed is needed; the digest depends on the Pillow build's bundled font.\n"
                "    image, qa = synthetic_diagram()\n"
                "    questions = [{{'question': q, 'options': options}} for q, options, _ in qa]\n"
                "    correct = [gold for _, _, gold in qa]\n"
                "    image_name = 'synthetic_plant_diagram_640x640.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'max_new_tokens': max_new_tokens, 'n_questions': len(questions), 'has_labels': correct is not None}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `answer` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, each question a non-empty string of at most "
                "`MAX_QUESTION_CHARS` characters (whitespace collapsed), `MIN_OPTIONS`..`MAX_OPTIONS` distinct non-empty options "
                "of at most `MAX_OPTION_CHARS` characters, and `max_new_tokens` in `[1, MAX_NEW_TOKENS]` — and returns an "
                "**input manifest** naming the schema (including the header-rendering preprocessing and the decoding rule), "
                "the input's observed mode and size, each checked question with its options and the exact prompt that will be "
                "rendered, the budget and the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show "
                "what rejection looks like, the cell also validates a question with a single option and records the pipeline's "
                "own error message as a finding. Inside the pipeline the image is converted to RGB, the prompt is rendered above "
                "it, and the composite is scaled to the patch budget; nothing else is dropped or altered. The pipeline cannot "
                "tell whether the image is a diagram or whether the question is answerable from it: that contract is the "
                "caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MAX_QUESTION_CHARS': MAX_QUESTION_CHARS, 'MIN_OPTIONS': MIN_OPTIONS, 'MAX_OPTIONS': MAX_OPTIONS, 'MAX_OPTION_CHARS': MAX_OPTION_CHARS, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING}}}})\n"
                "input_manifest = validate_inputs(image, questions, max_new_tokens=max_new_tokens, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, [{{'question': 'What is this?', 'options': ['a plant']}}])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'single-option-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Answer the questions and read the output correctly\n\n"
                "`answer` returns, per question, a dict with `answer` (the decoded text, stripped), `choice_index` (the "
                "zero-based option the normalised text equals, or `null` when it equals none — there is no fuzzy match), the "
                "checked `question` and `options`, the rendered `prompt`, `image_size`, `new_tokens`, a `truncated` flag that "
                "is true when the budget was exhausted, the generation settings and the model identity. **No score exists**: the "
                "answer is generated text with no probability and no correctness signal, and matching an option is not "
                "evidence that it is the right one. Greedy decoding is deterministic on a fixed device and dtype; CUDA kernel "
                "selection can change a token and therefore the rest of the answer, so GPU and CPU outputs need not match. Each "
                "call renders the prompt as a header and re-encodes the diagram, so cost is per question (about 2.1–2.4 s each "
                "on the reference CPU). As recorded in the model card, the repository's CPU smoke on this same drawing answered "
                "`root` to the label questions 1–4 whenever `root` was among the options, `sun` and `root` correctly for the "
                "sun and below-the-soil questions, and produced `root root (2) root root (3) root root (4) root` — matching no "
                "option — on a blank white image: the model always produces text, whether or not an answer exists."
            ),
            "code": (
                "import time\n\n"
                "results, seconds = [], []\n"
                "for entry in questions:\n"
                "    t0 = time.time()\n"
                "    results.append(pipe.answer(image, entry['question'], entry['options'], max_new_tokens=max_new_tokens))\n"
                "    seconds.append(round(time.time() - t0, 2))\n"
                "print({{'device': pipe.device, 'dtype': pipe.dtype, 'seconds_per_question': seconds, 'any_truncated': any(r['truncated'] for r in results), 'unmatched': sum(r['choice_index'] is None for r in results)}})\n"
                "for result in results:\n"
                "    matched = result['options'][result['choice_index']] if result['choice_index'] is not None else 'NO OPTION MATCHED'\n"
                "    print(f\"Q: {{result['question']}}  options={{result['options']}}\\n   A: {{result['answer']!r}} -> {{matched}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\")\n"
                "if any(r['truncated'] for r in results):\n"
                "    print('A budget was exhausted: that answer is incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: AI2D-style accuracy needs labelled multiple-choice questions on diagrams from the "
                "deployment domain, and this repository ships none (the AI2D benchmark is not bundled). When the correct option "
                "indices are supplied the report carries `accuracy` (the matched option equals the correct one; an answer that "
                "matches no option counts as wrong), `unmatched_rate`, a `chance` baseline (the mean of 1/options over the "
                "questions) and one entry per question, with the verdict `sample-sanity`. On the synthetic path those labels are "
                "facts **you drew yourself**, so the score proves only that the input contract, header rendering, forward pass, "
                "decoding and option matching round-trip — and the six recorded misses show what a wrong answer looks like in "
                "the report. On BYOD no correct options are known, the verdict is `not-measurable`, and the report states what "
                "would make the task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(results, correct, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'per_question', 'baselines')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:15}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for baseline in report['baselines']:\n"
                "    print(f\"{{baseline['id']:15}} {{baseline['value']:.3f}}  ({{baseline['note']}})\")\n"
                "for entry in report.get('per_question', []):\n"
                "    print(f\"  {{'OK  ' if entry['correct'] else 'MISS'}}  {{entry['question']}} -> {{entry['prediction']!r}} (correct: {{entry['correct_option']!r}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No correct options are known for these questions, so nothing is scored; read the answers against the diagram yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every result (question, options, prompt, answer, matched index, `new_tokens`, "
                "`truncated`, the budget), the evaluation report, the input manifest, the sample identity, digest and correct "
                "options, the notebook's source (repository, revision, embedded module digest, generator), the model identifier, "
                "the immutable model revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, "
                "device). The question/answer pairs are also written as CSV with explicit `image`, `question`, `options`, "
                "`answer`, `choice_index`, `new_tokens`, `truncated` columns, and an annotated PNG shows the diagram with the "
                "questions and answers printed in a panel beneath it for visual inspection (the model returns no location, so "
                "nothing is drawn on the diagram itself) — a supplement to, not a replacement for, the machine-readable files. "
                "No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "panel_height = 30 + 22 * len(results)\n"
                "annotated = Image.new('RGB', (max(image.width, 900), image.height + panel_height), 'white')\n"
                "annotated.paste(image.convert('RGB'), (0, 0))\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "draw.line([(0, image.height + 1), (annotated.width, image.height + 1)], fill=(120, 120, 120), width=2)\n"
                "panel_font = ImageFont.load_default(size=14)\n"
                "for index, result in enumerate(results):\n"
                "    matched = result['options'][result['choice_index']] if result['choice_index'] is not None else '-'\n"
                "    draw.text((20, image.height + 12 + 22 * index), f\"{{result['question']}}  ->  {{result['answer']}} [{{matched}}]\", fill=(40, 90, 220), font=panel_font)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'predictions': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'questions': questions, 'correct': correct}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_answers.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'question', 'options', 'answer', 'choice_index', 'new_tokens', 'truncated'])\n"
                "    for result in results:\n"
                "        writer.writerow([image_name, result['question'], ' | '.join(result['options']), result['answer'], result['choice_index'], result['new_tokens'], result['truncated']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The answers are the text the model generates after reading a diagram with the question and options printed above "
        "it; nothing in the output scores that text, the model returns no location or evidence, and it answers every question "
        "— including one about a blank image — with equal fluency, sometimes with text that matches no option. On the drawn "
        "plant the `accuracy` in the evaluation report compares the matched options with facts you drew yourself and the "
        "verdict is `sample-sanity`, which proves only that the input contract, header rendering, forward pass, decoding and "
        "option matching work (the repository's smoke run scored 4/10 against a chance baseline of 0.25, answering `root` to "
        "nearly every label question whenever `root` was an option — and `blossom` when the options were synonyms without it); "
        "they say nothing about textbook diagrams, photographs, charts, questions that need reasoning across parts, or "
        "non-English prompts, and a BYOD result is a single-diagram observation with the verdict `not-measurable`. **The model "
        "answers any question about any image** and stops only at end-of-sequence or the token budget: check `truncated` and "
        "`choice_index`, and treat a plausible option for an unanswerable question — or a `null` match — as the expected "
        "failure mode, not an exception. The answer also depends on the option set itself (removing `root` changed the answer "
        "to `leaf`), so the options are part of the request, not a neutral scoring key. The pipeline provides no OCR, no answer "
        "localisation, no open-ended answering, no benchmark evaluation and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** reorder or replace the options of a label question and watch the answer follow the option set; "
        "ask `What color is the flower?` with `pink` among the options (the smoke run said `green`); lower `max_new_tokens` to "
        "1 and watch `truncated` turn true; enable `USE_BYOD` with a textbook diagram you know, type your questions as "
        "`question | option | option | …`, then pass your own correct indices to `evaluation_report` to see the verdict switch "
        "to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/pix2struct-ai2d-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/pix2struct-ai2d-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/pix2struct-ai2d-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/pix2struct\n"
        "- Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding (Lee et al., 2022): https://arxiv.org/abs/2210.03347\n"
        "- A Diagram Is Worth A Dozen Images — the AI2D dataset (Kembhavi et al., 2016): https://arxiv.org/abs/1603.07396"
    ),
}
