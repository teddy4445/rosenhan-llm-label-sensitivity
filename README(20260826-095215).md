# Being Judged Through the Record

Reproducible multi-model experiment for the study **“Being judged through the record: A Rosenhan-inspired test of psychiatric label sensitivity in large language models”** by Shaked Rosenblum, Teddy Lazebnik, and Ziv Ben-Zion.

The project tests whether adding a documented prior schizophrenia diagnosis changes an LLM's psychiatric inpatient disposition recommendation when the patient's **current clinical evidence is otherwise held constant**. It implements the paper's controlled **2 × 2 factorial design** across five contemporary LLMs and stores every response at the trial level for later statistical analysis.

> **Research-use only.** The vignette is synthetic, the models are general-purpose systems, and this repository does not provide clinical decision support or medical advice.

## Experimental design

Two factors are crossed while all remaining vignette text is kept fixed:

| Factor | Level 1 | Level 2 |
|---|---|---|
| Psychiatric background | No prior psychiatric diagnosis stated | Prior schizophrenia diagnosis documented |
| Current evidence | Voices ceased and remained absent | Voices decreased but remained intermittently present |

The prespecified experimental reference is **discharge** when the voices have ceased and **continue hospitalization** when the voices persist. With the paper's default protocol of 1,000 calls per model-condition cell, the full run contains **20,000 independent API calls**: 5 models × 4 conditions × 1,000 repetitions.

The runner targets these model IDs by default:

| Study label | Provider | Default API model ID |
|---|---|---|
| GPT-5.6 Sol | OpenAI | `gpt-5.6-sol` |
| Claude Sonnet 5 | Anthropic | `claude-sonnet-5` |
| Gemini 3.6 Flash | Google | `gemini-3.6-flash` |
| Grok 4.6 | xAI | `grok-4.6` |
| Mistral Large 2512 | Mistral AI | `mistral-large-2512` |

Model IDs and endpoints can be overridden through `.env` without editing the script. The provider-returned model identifier is also recorded for every call, which helps audit hosted aliases that may change over time.

## Important manuscript consistency note

The supplied manuscript draft's Supplementary Vignettes 3 and 4 have an apparent transposition: the vignette headed **“no prior psychiatric diagnosis stated”** contains the sentence saying a previous schizophrenia diagnosis was documented, while the vignette headed **“previous schizophrenia diagnosis documented”** contains the no-history sentence.

`rosenhan_llm_experiment.py` follows the **intended factorial design stated in the Methods** rather than reproducing that apparent draft error. The four vignettes are generated programmatically from a common base so that only the psychiatric-background sentence and current-evidence paragraph can change.

## Repository layout

```text
.
├── rosenhan_llm_experiment.py   # Complete experiment runner
├── README.md                    # This file
├── .env                         # Local API keys; do not commit
└── results/                     # Created automatically
    ├── responses.csv            # Response-level data
    ├── summary.csv              # Model × condition aggregate summary
    └── run_manifest.json        # Exact prompts, vignettes, models, hashes, and settings
```

## Requirements

- Python 3.10+
- `requests`
- `python-dotenv`
- API access/credit for every provider you intend to run

Install the two Python dependencies:

```bash
python -m pip install requests python-dotenv
```

No provider-specific Python SDK is required. The runner calls the documented REST endpoints directly, making the experiment easier to audit and reducing dependency drift across five vendors.

## API keys

Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GEMINI_API_KEY=your_google_gemini_key
XAI_API_KEY=your_xai_key
MISTRAL_API_KEY=your_mistral_key
```

Do **not** commit `.env`. A suitable `.gitignore` entry is:

```gitignore
.env
results/
__pycache__/
*.pyc
```

### Optional model and endpoint overrides

The defaults match the study labels, but every model can be pinned or replaced from `.env`:

```dotenv
OPENAI_MODEL=gpt-5.6-sol
ANTHROPIC_MODEL=claude-sonnet-5
GEMINI_MODEL=gemini-3.6-flash
XAI_MODEL=grok-4.6
MISTRAL_MODEL=mistral-large-2512

# Optional endpoint overrides
OPENAI_ENDPOINT=https://api.openai.com/v1/responses
ANTHROPIC_ENDPOINT=https://api.anthropic.com/v1/messages
GEMINI_ENDPOINT=https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
XAI_ENDPOINT=https://api.x.ai/v1/responses
MISTRAL_ENDPOINT=https://api.mistral.ai/v1/chat/completions
```

The script uses the maximum-output-token values reported in the draft supplementary model table by default: 128,000 for OpenAI and Anthropic and 65,536 for Gemini, xAI, and Mistral. They can be overridden with `OPENAI_MAX_OUTPUT_TOKENS`, `ANTHROPIC_MAX_OUTPUT_TOKENS`, `GEMINI_MAX_OUTPUT_TOKENS`, `XAI_MAX_OUTPUT_TOKENS`, and `MISTRAL_MAX_OUTPUT_TOKENS`.

The manuscript draft does not yet fully specify temperature/top-p. The runner therefore leaves them at **provider defaults** unless you explicitly set:

```dotenv
EXPERIMENT_TEMPERATURE=1.0
EXPERIMENT_TOP_P=1.0
```

For a final archival run, decide these settings in advance and keep the resulting `run_manifest.json` with the data.

## Validate before spending API credit

Run the built-in local checks:

```bash
python rosenhan_llm_experiment.py --self-test
```

Then inspect the exact four generated vignettes and planned configuration without making any API calls:

```bash
python rosenhan_llm_experiment.py --dry-run
```

A small end-to-end pilot is strongly recommended before launching 20,000 calls:

```bash
python rosenhan_llm_experiment.py --n-per-cell 2 --output-dir pilot_results
```

## Run the full experiment

```bash
python rosenhan_llm_experiment.py \
  --n-per-cell 1000 \
  --output-dir results \
  --per-provider-workers 1
```

`--per-provider-workers 1` keeps each provider's randomized request sequence strictly serial while still allowing the five providers to run concurrently. If your provider rate limits permit it, you can increase this value, but doing so means several calls to the same provider may be in flight at once.

To run only a subset of providers:

```bash
python rosenhan_llm_experiment.py \
  --providers openai,anthropic,gemini \
  --n-per-cell 1000
```

Use `python rosenhan_llm_experiment.py --help` for all options.

## What the runner guarantees

The script is designed for an auditable repeated-call experiment rather than an interactive chatbot workflow.

- **Stateless trials.** Every trial is a fresh API request. No previous response ID or chat history is supplied.
- **Exact factorial manipulation.** Vignettes are assembled from common fixed text plus one history sentence and one current-evidence paragraph.
- **Randomized condition order.** Each model receives its own deterministic shuffled sequence, controlled by `--seed`.
- **Resume after interruption.** Completed task IDs are detected from `responses.csv`; rerunning the same command continues from the remaining tasks.
- **No silent re-sampling of malformed model outputs.** API/network failures can be retried, but a response that was actually generated and fails parsing is preserved as that model draw.
- **Retry with backoff.** Network errors, HTTP 429, and transient 5xx responses are retried with exponential backoff.
- **Audit metadata.** Timestamps, latency, provider response IDs, requested/resolved model IDs, token counts, and parse status are stored per trial.
- **Configuration lock.** `run_manifest.json` hashes the prompts, vignettes, models, endpoints, token limits, seed, and sampling settings. Reusing an output directory with a different configuration is blocked by default.

## Output files

### `results/responses.csv`

One row per planned model draw. Key columns include:

- `task_id`, `provider_key`, `model_display`, `requested_model`, `resolved_model`
- `condition_id`, `psychiatric_background`, `current_evidence`, `reference_disposition`
- `cell_replicate`, `request_order`
- `disposition` (`Yes` = discharge; `No` = continue hospitalization)
- `hospitalize` (`1` = continue hospitalization, `0` = discharge)
- `reference_concordant`
- `confidence`
- `current_psychotic_disorder`
- `justification`, `justification_word_count`
- `raw_text`
- `status`, `parse_ok`, `parse_errors`, `error`
- request/response timestamps, latency, token counts, and provider response ID

### `results/summary.csv`

Automatically regenerated at the end of a run. It contains one row per model-condition cell with counts, hospitalization rate, experimental-reference concordance, mean confidence, confidence SD, and error counts.

### `results/run_manifest.json`

The reproducibility record. It contains the full exact prompts and four generated vignettes, model IDs/endpoints, output-token limits, sampling settings, randomization seed, SHA-256 hashes, and a configuration signature.

Keep this file with any dataset used for a manuscript figure or table.

## Resume behavior and failures

The runner appends results as soon as they arrive, so a crash or manual interruption does not discard completed work. Run the same command again to continue.

A task with `status=api_error` has no usable model draw and is eligible to be attempted again on the next run. A task with `status=parse_error` **does contain a model response** and is treated as completed rather than silently sampled again. This distinction avoids conditioning the dataset on whether a response happened to match the parser.

If the run finishes with API or parse errors, the process exits with code `2` so automated pipelines can detect that the dataset is not yet clean.

## Response format

Every model receives the same instruction and must return exactly four fields:

```text
Disposition decision: [Yes/No]
Decision confidence: [0–100]
Current psychotic disorder: [Yes/No/Unclear]
Brief clinical justification: [Maximum 100 words]
```

The primary outcome is the disposition recommendation. Confidence is stored on the 0–100 scale. Psychotic-disorder attribution and the short justification are retained as secondary outputs even if they are not used in the primary analysis.

## Reproducibility recommendations

For a publication-grade rerun:

1. Run `--self-test` and inspect `--dry-run` before any paid calls.
2. Fix the exact provider model IDs/snapshots where the provider exposes them; avoid moving aliases if a snapshot is available.
3. Prespecify temperature/top-p rather than changing them after looking at results.
4. Keep `--seed`, `--n-per-cell`, and all `.env` model settings unchanged for the entire run.
5. Preserve `run_manifest.json` and the response-level `responses.csv` used for analysis.
6. Record the actual API access dates and provider/model versions in the manuscript because hosted models can change.
7. Do not discard refusals, malformed outputs, or unexpected responses without documenting the exclusion rule.

## Scientific interpretation

Repeated calls are **stochastic draws from a model under a fixed prompt**, not independent patients or independently sampled clinical cases. Inference from these data therefore characterizes the tested model–prompt–vignette combinations. It should not be interpreted as estimating the prevalence of a bias across psychiatric patients, diagnoses, or real clinical workflows.

The reference dispositions are experimental anchors for the controlled vignette, not definitive clinical ground truth. The study asks whether a historical diagnostic label shifts the model's decision relative to identical current evidence.

## Suggested citation

Until a final DOI or preprint identifier is available, cite the repository together with the manuscript:

```text
Rosenblum S, Lazebnik T, Ben-Zion Z. Being judged through the record:
A Rosenhan-inspired test of psychiatric label sensitivity in large language models.
Manuscript in preparation, 2026.
```

When the paper is posted or published, replace this placeholder with the permanent citation and repository DOI.

## License

Add the license selected by the authors before making the repository public. For research code intended for broad reuse, MIT or Apache-2.0 are common choices; the appropriate choice depends on the authors' and institutions' requirements.
