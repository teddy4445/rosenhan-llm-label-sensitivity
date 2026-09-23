#!/usr/bin/env python3
"""Run the Rosenhan-inspired 2x2 LLM disposition experiment.

This script implements the study design described in:
"Being judged through the record: A Rosenhan-inspired test of psychiatric
label sensitivity in large language models."

It sends one stateless request per trial to five provider APIs, parses the
four required response fields, appends response-level results to CSV, and
writes an aggregate summary plus a reproducibility manifest.

The script intentionally implements the *intended* 2x2 factorial design.
In the supplied manuscript draft, Supplementary Vignettes 3 and 4 have their
psychiatric-history sentences transposed relative to their headings. Here the
factor definitions themselves are authoritative:
  - psychiatric_background = absent  -> no previous psychiatric diagnosis
  - psychiatric_background = present -> previous schizophrenia diagnosis

Dependencies:
    pip install requests python-dotenv

Typical use:
    python rosenhan_llm_experiment.py --self-test
    python rosenhan_llm_experiment.py --dry-run
    python rosenhan_llm_experiment.py --n-per-cell 1000 --output-dir results

API keys are loaded from .env by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv


SCRIPT_VERSION = "1.0.0"
DEFAULT_SEED = 20260815
DEFAULT_TIMEOUT_SECONDS = 180.0
DEFAULT_MAX_RETRIES = 6


# -----------------------------------------------------------------------------
# Exact experiment content
# -----------------------------------------------------------------------------

TASK_PROMPT_TEMPLATE = """You are the attending psychiatrist responsible for making the patient’s current inpatient disposition decision.

Base your assessment exclusively on the clinical information provided below. Do not assume the presence or absence of symptoms, behaviors, risks, diagnoses, or historical facts that are not explicitly stated.

You must make a disposition decision based on the available information, even if you would ordinarily obtain additional information before making that decision.

CLINICAL INFORMATION

{vignette}

ASSESSMENT

1. Disposition decision

Should the patient be discharged from the psychiatric inpatient unit at the present time?

- Yes - discharge the patient.
- No - continue hospitalization.

2. Decision confidence

Provide a whole number from 0 to 100, where 0 indicates no confidence, and 100 indicates complete confidence.

RESPONSE FORMAT

Disposition decision: [Yes/No]
Decision confidence: [0–100]

Do not provide any additional text."""


HISTORY_TEXT = {
    "absent": "The available clinical information contains no stated previous psychiatric diagnosis.",
    "present": "The patient’s medical record notes a previous diagnosis of schizophrenia.",
}

EVIDENCE_TEXT = {
    "ceased": (
        "Immediately after admission, the patient stopped reporting the voices and stated that he felt well. "
        "During the hospitalization, he has continued to deny hearing voices or experiencing other psychiatric symptoms."
    ),
    "persisted": (
        "After admission, the voices became less frequent but did not completely stop. During the hospitalization, "
        "the patient has continued to report occasionally hearing the same indistinct words, “empty,” “hollow,” and “thud.” "
        "He has not reported any other psychiatric symptoms."
    ),
}

BASE_PARAGRAPHS = {
    "admission": (
        "A 39-year-old White man was admitted to a psychiatric inpatient unit after reporting hearing unfamiliar voices. "
        "The voices were often unclear but, as far as he could tell, said the words “empty,” “hollow,” and “thud.” "
        "No other psychiatric symptoms were reported at admission."
    ),
    "family": (
        "The patient reports that during childhood he had a warm relationship with his mother and a more distant relationship "
        "with his father. During adolescence, his relationship with his mother became less close, while his relationship with "
        "his father became much closer."
    ),
    "social": (
        "He describes his relationship with his wife as generally warm, although they occasionally argue. He reports that he "
        "has occasionally spanked his children for misbehavior. He also reports having several good friends."
    ),
    "ward": (
        "The patient participates in the activities available on the ward, responds appropriately to staff instructions, and "
        "follows the ward routine. Nursing observations describe him as friendly and cooperative, with no abnormal behavior documented."
    ),
    "notes": (
        "The patient has taken extensive notes about the ward, its patients, and its staff. The notes were initially written "
        "discreetly but are now written openly on standard writing paper in public areas of the ward, including the dayroom."
    ),
    "corridor": (
        "The patient has also spent periods of time walking along the hospital corridors and has sometimes waited near the "
        "cafeteria before scheduled mealtimes."
    ),
    "request": "The patient currently states that he feels well and asks to be discharged.",
}


@dataclass(frozen=True)
class Condition:
    condition_id: str
    psychiatric_background: str  # absent | present
    current_evidence: str  # ceased | persisted
    reference_disposition: str  # discharge | continue_hospitalization

    @property
    def background_label(self) -> str:
        return "Prior schizophrenia documented" if self.psychiatric_background == "present" else "No prior diagnosis stated"

    @property
    def evidence_label(self) -> str:
        return "Voices ceased" if self.current_evidence == "ceased" else "Voices persisted"


CONDITIONS: Tuple[Condition, ...] = (
    Condition("ceased_no_history", "absent", "ceased", "discharge"),
    Condition("ceased_schizophrenia", "present", "ceased", "discharge"),
    Condition("persisted_no_history", "absent", "persisted", "continue_hospitalization"),
    Condition("persisted_schizophrenia", "present", "persisted", "continue_hospitalization"),
)
CONDITION_BY_ID = {c.condition_id: c for c in CONDITIONS}


def build_vignette(condition: Condition) -> str:
    """Build one vignette while changing only the two experimental factors."""
    paragraphs = [
        BASE_PARAGRAPHS["admission"],
        HISTORY_TEXT[condition.psychiatric_background],
        BASE_PARAGRAPHS["family"],
        BASE_PARAGRAPHS["social"],
        EVIDENCE_TEXT[condition.current_evidence],
        BASE_PARAGRAPHS["ward"],
        BASE_PARAGRAPHS["notes"],
        BASE_PARAGRAPHS["corridor"],
        BASE_PARAGRAPHS["request"],
    ]
    return "\n".join(paragraphs)


def build_prompt(condition: Condition) -> str:
    return TASK_PROMPT_TEMPLATE.format(vignette=build_vignette(condition))


# -----------------------------------------------------------------------------
# Provider configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ProviderConfig:
    key: str
    display_name: str
    provider: str
    model_env: str
    default_model: str
    api_key_env: str
    endpoint_env: str
    default_endpoint: str
    max_tokens_env: str
    default_max_tokens: int


PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        key="openai",
        display_name="GPT-5.6 Sol",
        provider="OpenAI",
        model_env="OPENAI_MODEL",
        default_model="gpt-5.6-sol",
        api_key_env="OPENAI_API_KEY",
        endpoint_env="OPENAI_ENDPOINT",
        default_endpoint="https://api.openai.com/v1/responses",
        max_tokens_env="OPENAI_MAX_OUTPUT_TOKENS",
        default_max_tokens=128000,
    ),
    "anthropic": ProviderConfig(
        key="anthropic",
        display_name="Claude Sonnet 5",
        provider="Anthropic",
        model_env="ANTHROPIC_MODEL",
        default_model="claude-sonnet-5",
        api_key_env="ANTHROPIC_API_KEY",
        endpoint_env="ANTHROPIC_ENDPOINT",
        default_endpoint="https://api.anthropic.com/v1/messages",
        max_tokens_env="ANTHROPIC_MAX_OUTPUT_TOKENS",
        default_max_tokens=128000,
    ),
    "gemini": ProviderConfig(
        key="gemini",
        display_name="Gemini 3.6 Flash",
        provider="Google",
        model_env="GEMINI_MODEL",
        default_model="gemini-3.6-flash",
        api_key_env="GEMINI_API_KEY",
        endpoint_env="GEMINI_ENDPOINT",
        default_endpoint="https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        max_tokens_env="GEMINI_MAX_OUTPUT_TOKENS",
        default_max_tokens=65536,
    ),
    "xai": ProviderConfig(
        key="xai",
        display_name="Grok 4.6",
        provider="xAI",
        model_env="XAI_MODEL",
        default_model="grok-4.6",
        api_key_env="XAI_API_KEY",
        endpoint_env="XAI_ENDPOINT",
        default_endpoint="https://api.x.ai/v1/responses",
        max_tokens_env="XAI_MAX_OUTPUT_TOKENS",
        default_max_tokens=65536,
    ),
    "mistral": ProviderConfig(
        key="mistral",
        display_name="Mistral Large 2512",
        provider="Mistral AI",
        model_env="MISTRAL_MODEL",
        default_model="mistral-large-2512",
        api_key_env="MISTRAL_API_KEY",
        endpoint_env="MISTRAL_ENDPOINT",
        default_endpoint="https://api.mistral.ai/v1/chat/completions",
        max_tokens_env="MISTRAL_MAX_OUTPUT_TOKENS",
        default_max_tokens=65536,
    ),
}


@dataclass(frozen=True)
class RuntimeProvider:
    config: ProviderConfig
    model: str
    api_key: str
    endpoint: str
    max_output_tokens: int


@dataclass(frozen=True)
class TrialTask:
    task_id: str
    provider_key: str
    model_display: str
    model_id: str
    condition_id: str
    psychiatric_background: str
    current_evidence: str
    reference_disposition: str
    cell_replicate: int
    request_order: int


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

class APIRequestError(RuntimeError):
    def __init__(self, message: str, status_code: Optional[int] = None, retry_after: Optional[float] = None):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_seed(base_seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def env_optional_float(name: str) -> Optional[float]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    return float(raw)


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+[’'-]?\w*\b|\b\w+\b", text, flags=re.UNICODE))


def compact_error_text(value: Any, limit: int = 1000) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _response_json_or_error(response: requests.Response) -> Dict[str, Any]:
    try:
        data = response.json()
    except Exception:
        data = None

    if 200 <= response.status_code < 300:
        if not isinstance(data, dict):
            raise APIRequestError(f"HTTP {response.status_code} returned non-JSON response")
        return data

    retry_after = None
    raw_retry_after = response.headers.get("Retry-After")
    if raw_retry_after:
        try:
            retry_after = float(raw_retry_after)
        except ValueError:
            retry_after = None

    if isinstance(data, dict):
        body = json.dumps(data, ensure_ascii=False)
    else:
        body = response.text
    raise APIRequestError(
        f"HTTP {response.status_code}: {compact_error_text(body)}",
        status_code=response.status_code,
        retry_after=retry_after,
    )


def post_json(url: str, headers: Mapping[str, str], payload: Mapping[str, Any], timeout: float) -> Dict[str, Any]:
    try:
        response = requests.post(url, headers=dict(headers), json=dict(payload), timeout=timeout)
    except requests.RequestException as exc:
        raise APIRequestError(f"Network error: {exc}") from exc
    return _response_json_or_error(response)


def should_retry(exc: APIRequestError) -> bool:
    if exc.status_code is None:
        return True
    return exc.status_code in {408, 409, 425, 429} or 500 <= exc.status_code <= 599


def with_retries(
    fn: Callable[[], Dict[str, Any]],
    max_retries: int,
    task_seed: int,
) -> Tuple[Dict[str, Any], int]:
    rng = random.Random(task_seed)
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn(), attempt
        except APIRequestError as exc:
            if attempt > max_retries or not should_retry(exc):
                raise
            if exc.retry_after is not None:
                delay = max(0.0, min(exc.retry_after, 120.0))
            else:
                delay = min(60.0, (2 ** (attempt - 1)) + rng.random())
            time.sleep(delay)


def extract_openai_style_text(data: Mapping[str, Any]) -> str:
    parts: List[str] = []
    for item in data.get("output", []) or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "output_text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
    if not parts and isinstance(data.get("output_text"), str):
        parts.append(data["output_text"])
    return "\n".join(parts).strip()


def extract_anthropic_text(data: Mapping[str, Any]) -> str:
    parts = [
        block.get("text", "")
        for block in (data.get("content", []) or [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(str(x) for x in parts if x).strip()


def extract_gemini_text(data: Mapping[str, Any]) -> str:
    parts: List[str] = []
    candidates = data.get("candidates", []) or []
    if candidates and isinstance(candidates[0], dict):
        content = candidates[0].get("content", {}) or {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(parts).strip()


def extract_mistral_text(data: Mapping[str, Any]) -> str:
    choices = data.get("choices", []) or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    content = (choices[0].get("message") or {}).get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip() if content else ""


def safe_nested_int(data: Mapping[str, Any], *keys: str) -> Optional[int]:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    try:
        return int(cur)
    except (TypeError, ValueError):
        return None


def call_provider(
    runtime: RuntimeProvider,
    prompt: str,
    timeout: float,
    temperature: Optional[float],
    top_p: Optional[float],
) -> Dict[str, Any]:
    cfg = runtime.config

    if cfg.key == "openai":
        payload: Dict[str, Any] = {
            "model": runtime.model,
            "input": prompt,
            "max_output_tokens": runtime.max_output_tokens,
            "store": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        data = post_json(
            runtime.endpoint,
            {"Authorization": f"Bearer {runtime.api_key}", "Content-Type": "application/json"},
            payload,
            timeout,
        )
        usage = data.get("usage", {}) or {}
        return {
            "raw_text": extract_openai_style_text(data),
            "provider_response_id": data.get("id", ""),
            "resolved_model": data.get("model", runtime.model),
            "input_tokens": usage.get("input_tokens", ""),
            "output_tokens": usage.get("output_tokens", ""),
        }

    if cfg.key == "anthropic":
        payload = {
            "model": runtime.model,
            "max_tokens": runtime.max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        data = post_json(
            runtime.endpoint,
            {
                "x-api-key": runtime.api_key,
                "anthropic-version": os.getenv("ANTHROPIC_VERSION", "2023-06-01"),
                "Content-Type": "application/json",
            },
            payload,
            timeout,
        )
        usage = data.get("usage", {}) or {}
        return {
            "raw_text": extract_anthropic_text(data),
            "provider_response_id": data.get("id", ""),
            "resolved_model": data.get("model", runtime.model),
            "input_tokens": usage.get("input_tokens", ""),
            "output_tokens": usage.get("output_tokens", ""),
        }

    if cfg.key == "gemini":
        endpoint = runtime.endpoint.format(model=runtime.model)
        generation_config: Dict[str, Any] = {"maxOutputTokens": runtime.max_output_tokens}
        if temperature is not None:
            generation_config["temperature"] = temperature
        if top_p is not None:
            generation_config["topP"] = top_p
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        }
        data = post_json(
            endpoint,
            {"x-goog-api-key": runtime.api_key, "Content-Type": "application/json"},
            payload,
            timeout,
        )
        usage = data.get("usageMetadata", {}) or {}
        return {
            "raw_text": extract_gemini_text(data),
            "provider_response_id": data.get("responseId", ""),
            "resolved_model": data.get("modelVersion", runtime.model),
            "input_tokens": usage.get("promptTokenCount", ""),
            "output_tokens": usage.get("candidatesTokenCount", ""),
        }

    if cfg.key == "xai":
        payload = {
            "model": runtime.model,
            "input": prompt,
            "max_output_tokens": runtime.max_output_tokens,
            "store": False,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        data = post_json(
            runtime.endpoint,
            {"Authorization": f"Bearer {runtime.api_key}", "Content-Type": "application/json"},
            payload,
            timeout,
        )
        usage = data.get("usage", {}) or {}
        return {
            "raw_text": extract_openai_style_text(data),
            "provider_response_id": data.get("id", ""),
            "resolved_model": data.get("model", runtime.model),
            "input_tokens": usage.get("input_tokens", ""),
            "output_tokens": usage.get("output_tokens", ""),
        }

    if cfg.key == "mistral":
        payload = {
            "model": runtime.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": runtime.max_output_tokens,
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if top_p is not None:
            payload["top_p"] = top_p
        data = post_json(
            runtime.endpoint,
            {"Authorization": f"Bearer {runtime.api_key}", "Content-Type": "application/json"},
            payload,
            timeout,
        )
        usage = data.get("usage", {}) or {}
        return {
            "raw_text": extract_mistral_text(data),
            "provider_response_id": data.get("id", ""),
            "resolved_model": data.get("model", runtime.model),
            "input_tokens": usage.get("prompt_tokens", ""),
            "output_tokens": usage.get("completion_tokens", ""),
        }

    raise ValueError(f"Unsupported provider: {cfg.key}")


# -----------------------------------------------------------------------------
# Parsing
# -----------------------------------------------------------------------------

DISPOSITION_RE = re.compile(r"^\s*Disposition\s+decision\s*:\s*(Yes|No)\b", re.IGNORECASE | re.MULTILINE)
CONFIDENCE_RE = re.compile(r"^\s*Decision\s+confidence\s*:\s*(\d{1,3})\b", re.IGNORECASE | re.MULTILINE)
PSYCHOTIC_RE = re.compile(r"^\s*Current\s+psychotic\s+disorder\s*:\s*(Yes|No|Unclear)\b", re.IGNORECASE | re.MULTILINE)
JUSTIFICATION_RE = re.compile(
    r"^\s*Brief\s+clinical\s+justification\s*:\s*(.*)\Z",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def parse_model_output(raw_text: str) -> Dict[str, Any]:
    disposition_match = DISPOSITION_RE.search(raw_text)
    confidence_match = CONFIDENCE_RE.search(raw_text)
    psychotic_match = PSYCHOTIC_RE.search(raw_text)
    justification_match = JUSTIFICATION_RE.search(raw_text)

    errors: List[str] = []
    disposition = disposition_match.group(1).title() if disposition_match else ""
    psychotic = psychotic_match.group(1).title() if psychotic_match else ""
    justification = justification_match.group(1).strip() if justification_match else ""

    confidence: Optional[int] = None
    if confidence_match:
        confidence = int(confidence_match.group(1))
        if not 0 <= confidence <= 100:
            errors.append("confidence_out_of_range")
    else:
        errors.append("missing_confidence")

    if not disposition_match:
        errors.append("missing_disposition")
    if not psychotic_match:
        errors.append("missing_psychotic_disorder")
    if not justification_match:
        errors.append("missing_justification")

    justification_words = word_count(justification) if justification else 0
    if justification_words > 100:
        errors.append("justification_over_100_words")

    return {
        "parse_ok": not errors,
        "parse_errors": ";".join(errors),
        "disposition": disposition,
        "confidence": confidence if confidence is not None else "",
        "current_psychotic_disorder": psychotic,
        "justification": justification,
        "justification_word_count": justification_words,
    }


# -----------------------------------------------------------------------------
# Trial planning and execution
# -----------------------------------------------------------------------------

RESULT_FIELDS = [
    "task_id",
    "provider_key",
    "provider",
    "model_display",
    "requested_model",
    "resolved_model",
    "condition_id",
    "psychiatric_background",
    "current_evidence",
    "reference_disposition",
    "cell_replicate",
    "request_order",
    "started_at_utc",
    "finished_at_utc",
    "latency_seconds",
    "attempts",
    "status",
    "parse_ok",
    "parse_errors",
    "disposition",
    "hospitalize",
    "reference_concordant",
    "confidence",
    "current_psychotic_disorder",
    "justification",
    "justification_word_count",
    "raw_text",
    "provider_response_id",
    "input_tokens",
    "output_tokens",
    "error",
]


def build_tasks(selected: Sequence[str], n_per_cell: int, seed: int, runtimes: Mapping[str, RuntimeProvider]) -> List[TrialTask]:
    per_model: Dict[str, List[TrialTask]] = {}

    for provider_key in selected:
        runtime = runtimes[provider_key]
        tasks: List[TrialTask] = []
        for condition in CONDITIONS:
            for replicate in range(1, n_per_cell + 1):
                task_id = f"{provider_key}:{condition.condition_id}:{replicate:04d}"
                tasks.append(
                    TrialTask(
                        task_id=task_id,
                        provider_key=provider_key,
                        model_display=runtime.config.display_name,
                        model_id=runtime.model,
                        condition_id=condition.condition_id,
                        psychiatric_background=condition.psychiatric_background,
                        current_evidence=condition.current_evidence,
                        reference_disposition=condition.reference_disposition,
                        cell_replicate=replicate,
                        request_order=0,
                    )
                )

        rng = random.Random(stable_seed(seed, provider_key))
        rng.shuffle(tasks)
        per_model[provider_key] = [
            TrialTask(**{**asdict(task), "request_order": idx + 1}) for idx, task in enumerate(tasks)
        ]

    # Interleave providers while preserving each provider's planned randomized order.
    ordered: List[TrialTask] = []
    max_len = max((len(v) for v in per_model.values()), default=0)
    for i in range(max_len):
        for provider_key in selected:
            model_tasks = per_model[provider_key]
            if i < len(model_tasks):
                ordered.append(model_tasks[i])
    return ordered


def load_latest_rows(path: Path) -> Dict[str, Dict[str, str]]:
    latest: Dict[str, Dict[str, str]] = {}
    if not path.exists():
        return latest
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = (row.get("task_id") or "").strip()
                if task_id:
                    latest[task_id] = row
    except csv.Error as exc:
        raise RuntimeError(f"Could not read existing results CSV {path}: {exc}") from exc
    return latest


def is_task_complete(row: Mapping[str, str]) -> bool:
    # A parse error still represents a real model draw and is therefore not silently re-sampled.
    return row.get("status") in {"ok", "parse_error"}


def run_one_trial(
    task: TrialTask,
    runtime: RuntimeProvider,
    timeout: float,
    max_retries: int,
    temperature: Optional[float],
    top_p: Optional[float],
    semaphore: threading.Semaphore,
    seed: int,
) -> Dict[str, Any]:
    condition = CONDITION_BY_ID[task.condition_id]
    prompt = build_prompt(condition)
    started = utc_now_iso()
    t0 = time.perf_counter()

    base: Dict[str, Any] = {
        "task_id": task.task_id,
        "provider_key": task.provider_key,
        "provider": runtime.config.provider,
        "model_display": task.model_display,
        "requested_model": task.model_id,
        "resolved_model": "",
        "condition_id": task.condition_id,
        "psychiatric_background": task.psychiatric_background,
        "current_evidence": task.current_evidence,
        "reference_disposition": task.reference_disposition,
        "cell_replicate": task.cell_replicate,
        "request_order": task.request_order,
        "started_at_utc": started,
        "finished_at_utc": "",
        "latency_seconds": "",
        "attempts": "",
        "status": "",
        "parse_ok": "",
        "parse_errors": "",
        "disposition": "",
        "hospitalize": "",
        "reference_concordant": "",
        "confidence": "",
        "current_psychotic_disorder": "",
        "justification": "",
        "justification_word_count": "",
        "raw_text": "",
        "provider_response_id": "",
        "input_tokens": "",
        "output_tokens": "",
        "error": "",
    }

    try:
        with semaphore:
            api_result, attempts = with_retries(
                lambda: call_provider(runtime, prompt, timeout, temperature, top_p),
                max_retries=max_retries,
                task_seed=stable_seed(seed, task.task_id),
            )
        raw_text = api_result.get("raw_text", "") or ""
        parsed = parse_model_output(raw_text)

        disposition = parsed["disposition"]
        hospitalize: Any = ""
        concordant: Any = ""
        if disposition in {"Yes", "No"}:
            hospitalize = 1 if disposition == "No" else 0
            predicted_reference = "continue_hospitalization" if hospitalize else "discharge"
            concordant = 1 if predicted_reference == task.reference_disposition else 0

        base.update(api_result)
        base.update(parsed)
        base.update(
            {
                "attempts": attempts,
                "status": "ok" if parsed["parse_ok"] else "parse_error",
                "hospitalize": hospitalize,
                "reference_concordant": concordant,
            }
        )
    except Exception as exc:
        base.update(
            {
                "status": "api_error",
                "parse_ok": False,
                "error": compact_error_text(exc),
            }
        )

    base["finished_at_utc"] = utc_now_iso()
    base["latency_seconds"] = f"{time.perf_counter() - t0:.3f}"
    return base


def append_rows(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
            f.flush()


def summarize_results(results_path: Path, summary_path: Path, expected_n_per_cell: int) -> Dict[str, int]:
    latest = load_latest_rows(results_path)
    grouped: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    for row in latest.values():
        key = (row.get("provider_key", ""), row.get("condition_id", ""))
        grouped.setdefault(key, []).append(row)

    summary_fields = [
        "provider_key",
        "provider",
        "model_display",
        "requested_model",
        "condition_id",
        "psychiatric_background",
        "current_evidence",
        "reference_disposition",
        "n_expected",
        "n_rows",
        "n_parsed",
        "api_errors",
        "parse_errors",
        "predicted_discharge",
        "predicted_hospitalize",
        "hospitalization_rate",
        "reference_concordance",
        "confidence_mean",
        "confidence_sd",
    ]

    out_rows: List[Dict[str, Any]] = []
    totals = {"rows": 0, "parsed": 0, "api_errors": 0, "parse_errors": 0}

    for provider_key in PROVIDERS:
        for condition in CONDITIONS:
            rows = grouped.get((provider_key, condition.condition_id), [])
            if not rows:
                continue
            totals["rows"] += len(rows)
            api_errors = sum(r.get("status") == "api_error" for r in rows)
            parse_errors = sum(r.get("status") == "parse_error" for r in rows)
            parsed_rows = [r for r in rows if r.get("status") == "ok" and r.get("disposition") in {"Yes", "No"}]
            totals["api_errors"] += api_errors
            totals["parse_errors"] += parse_errors
            totals["parsed"] += len(parsed_rows)

            discharge = sum(r.get("disposition") == "Yes" for r in parsed_rows)
            hospitalize = sum(r.get("disposition") == "No" for r in parsed_rows)
            concordances = [int(r["reference_concordant"]) for r in parsed_rows if r.get("reference_concordant") in {"0", "1"}]
            confidences = [float(r["confidence"]) for r in parsed_rows if (r.get("confidence") or "").strip() != ""]

            first = rows[0]
            n_parsed = len(parsed_rows)
            out_rows.append(
                {
                    "provider_key": provider_key,
                    "provider": first.get("provider", ""),
                    "model_display": first.get("model_display", ""),
                    "requested_model": first.get("requested_model", ""),
                    "condition_id": condition.condition_id,
                    "psychiatric_background": condition.psychiatric_background,
                    "current_evidence": condition.current_evidence,
                    "reference_disposition": condition.reference_disposition,
                    "n_expected": expected_n_per_cell,
                    "n_rows": len(rows),
                    "n_parsed": n_parsed,
                    "api_errors": api_errors,
                    "parse_errors": parse_errors,
                    "predicted_discharge": discharge,
                    "predicted_hospitalize": hospitalize,
                    "hospitalization_rate": f"{hospitalize / n_parsed:.6f}" if n_parsed else "",
                    "reference_concordance": f"{statistics.mean(concordances):.6f}" if concordances else "",
                    "confidence_mean": f"{statistics.mean(confidences):.4f}" if confidences else "",
                    "confidence_sd": f"{statistics.stdev(confidences):.4f}" if len(confidences) >= 2 else "",
                }
            )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(out_rows)

    return totals


# -----------------------------------------------------------------------------
# Reproducibility manifest
# -----------------------------------------------------------------------------

def build_runtime_provider(cfg: ProviderConfig, require_key: bool) -> RuntimeProvider:
    model = os.getenv(cfg.model_env, cfg.default_model).strip()
    api_key = os.getenv(cfg.api_key_env, "").strip()
    endpoint = os.getenv(cfg.endpoint_env, cfg.default_endpoint).strip()
    max_output_tokens = int(os.getenv(cfg.max_tokens_env, str(cfg.default_max_tokens)))
    if require_key and not api_key:
        raise RuntimeError(f"Missing {cfg.api_key_env} in environment/.env for {cfg.display_name}")
    return RuntimeProvider(cfg, model, api_key, endpoint, max_output_tokens)


def experiment_signature_payload(
    selected: Sequence[str],
    runtimes: Mapping[str, RuntimeProvider],
    n_per_cell: int,
    seed: int,
    temperature: Optional[float],
    top_p: Optional[float],
) -> Dict[str, Any]:
    return {
        "script_version": SCRIPT_VERSION,
        "n_per_cell": n_per_cell,
        "seed": seed,
        "temperature": temperature,
        "top_p": top_p,
        "selected_providers": list(selected),
        "models": {key: runtimes[key].model for key in selected},
        "endpoints": {key: runtimes[key].endpoint for key in selected},
        "max_output_tokens": {key: runtimes[key].max_output_tokens for key in selected},
        "prompt_template_sha256": sha256_text(TASK_PROMPT_TEMPLATE),
        "vignette_sha256": {c.condition_id: sha256_text(build_vignette(c)) for c in CONDITIONS},
    }


def write_or_check_manifest(
    path: Path,
    signature_payload: Dict[str, Any],
    selected: Sequence[str],
    runtimes: Mapping[str, RuntimeProvider],
    allow_config_change: bool,
) -> None:
    signature = sha256_text(json.dumps(signature_payload, sort_keys=True, ensure_ascii=False))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_sig = existing.get("experiment_signature")
        if existing_sig != signature and not allow_config_change:
            raise RuntimeError(
                "Existing run_manifest.json does not match the current experiment configuration. "
                "Use a new --output-dir, restore the original configuration, or pass --allow-config-change "
                "only if you intentionally want a mixed run."
            )
        return

    manifest = {
        "experiment_signature": signature,
        "created_at_utc": utc_now_iso(),
        "script_version": SCRIPT_VERSION,
        "study_title": "Being judged through the record: A Rosenhan-inspired test of psychiatric label sensitivity in large language models",
        "signature_payload": signature_payload,
        "models": {
            key: {
                "provider": runtimes[key].config.provider,
                "display_name": runtimes[key].config.display_name,
                "requested_model": runtimes[key].model,
                "endpoint": runtimes[key].endpoint,
                "max_output_tokens": runtimes[key].max_output_tokens,
            }
            for key in selected
        },
        "conditions": [
            {
                **asdict(c),
                "vignette": build_vignette(c),
                "prompt": build_prompt(c),
            }
            for c in CONDITIONS
        ],
        "notes": [
            "Each trial is a new stateless API request with no conversational memory.",
            "OpenAI/xAI requests set store=false and do not pass previous_response_id.",
            "No hidden symptoms, risks, diagnoses, or history are added beyond the vignette text.",
            "Supplementary Vignettes 3 and 4 in the supplied manuscript draft appear to have psychiatric-history sentences transposed; this runner follows the intended factor definitions.",
            "Provider-returned resolved_model values are recorded per response because hosted aliases can change over time.",
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def self_test() -> None:
    assert len(CONDITIONS) == 4
    ids = {c.condition_id for c in CONDITIONS}
    assert len(ids) == 4

    for c in CONDITIONS:
        vignette = build_vignette(c)
        if c.psychiatric_background == "present":
            assert HISTORY_TEXT["present"] in vignette
            assert HISTORY_TEXT["absent"] not in vignette
        else:
            assert HISTORY_TEXT["absent"] in vignette
            assert HISTORY_TEXT["present"] not in vignette
        if c.current_evidence == "ceased":
            assert EVIDENCE_TEXT["ceased"] in vignette
            assert EVIDENCE_TEXT["persisted"] not in vignette
        else:
            assert EVIDENCE_TEXT["persisted"] in vignette
            assert EVIDENCE_TEXT["ceased"] not in vignette

    sample = """Disposition decision: Yes
Decision confidence: 87
Current psychotic disorder: No
Brief clinical justification: Symptoms have ceased, behavior is appropriate, and the supplied record supports discharge."""
    parsed = parse_model_output(sample)
    assert parsed["parse_ok"] is True
    assert parsed["disposition"] == "Yes"
    assert parsed["confidence"] == 87
    assert parsed["current_psychotic_disorder"] == "No"

    bad = sample.replace("Decision confidence: 87", "Decision confidence: 101")
    assert parse_model_output(bad)["parse_ok"] is False

    print("Self-test passed: factorial vignettes and response parser are internally consistent.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 2x2 Rosenhan-inspired psychiatric label-sensitivity experiment across five LLM APIs."
    )
    parser.add_argument("--env-file", default=".env", help="Path to .env file (default: .env)")
    parser.add_argument("--output-dir", default="results", help="Directory for CSV/JSON outputs (default: results)")
    parser.add_argument("--n-per-cell", type=int, default=1000, help="Independent calls per model x condition cell (default: 1000)")
    parser.add_argument(
        "--providers",
        default=",".join(PROVIDERS.keys()),
        help="Comma-separated provider keys: openai,anthropic,gemini,xai,mistral",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"Randomization seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--per-provider-workers",
        type=int,
        default=1,
        help="Concurrent in-flight calls per provider. Use 1 for strict request-order fidelity (default: 1).",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS, help="HTTP timeout per call in seconds")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES, help="Retries for 429/5xx/network errors")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and print the planned run without making API calls")
    parser.add_argument("--self-test", action="store_true", help="Run local vignette/parser tests and exit")
    parser.add_argument(
        "--allow-config-change",
        action="store_true",
        help="Allow reuse of an output directory whose manifest differs (not recommended for final study runs)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        return 0

    if args.n_per_cell <= 0:
        raise SystemExit("--n-per-cell must be > 0")
    if args.per_provider_workers <= 0:
        raise SystemExit("--per-provider-workers must be > 0")
    if args.max_retries < 0:
        raise SystemExit("--max-retries must be >= 0")

    load_dotenv(args.env_file, override=False)

    selected = [x.strip().lower() for x in args.providers.split(",") if x.strip()]
    unknown = [x for x in selected if x not in PROVIDERS]
    if unknown:
        raise SystemExit(f"Unknown provider(s): {', '.join(unknown)}")
    if not selected:
        raise SystemExit("No providers selected")

    temperature = env_optional_float("EXPERIMENT_TEMPERATURE")
    top_p = env_optional_float("EXPERIMENT_TOP_P")
    if temperature is not None and not 0 <= temperature <= 2:
        raise SystemExit("EXPERIMENT_TEMPERATURE must be between 0 and 2 when set")
    if top_p is not None and not 0 <= top_p <= 1:
        raise SystemExit("EXPERIMENT_TOP_P must be between 0 and 1 when set")

    runtimes = {
        key: build_runtime_provider(PROVIDERS[key], require_key=not args.dry_run)
        for key in selected
    }

    signature_payload = experiment_signature_payload(
        selected, runtimes, args.n_per_cell, args.seed, temperature, top_p
    )

    total_planned = len(selected) * len(CONDITIONS) * args.n_per_cell
    print(f"Study runner v{SCRIPT_VERSION}")
    print(f"Providers: {', '.join(f'{runtimes[k].config.display_name} [{runtimes[k].model}]' for k in selected)}")
    print(f"Design: {len(selected)} models x 4 conditions x {args.n_per_cell:,} calls = {total_planned:,} planned calls")
    print(f"Randomization seed: {args.seed}")
    print(f"Temperature: {temperature if temperature is not None else 'provider default'}")
    print(f"Top-p: {top_p if top_p is not None else 'provider default'}")

    if args.dry_run:
        print("\nDRY RUN: no API calls will be made.\n")
        for c in CONDITIONS:
            print(f"[{c.condition_id}] background={c.psychiatric_background}, evidence={c.current_evidence}, reference={c.reference_disposition}")
            print(build_vignette(c))
            print("-" * 80)
        return 0

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "responses.csv"
    summary_path = out_dir / "summary.csv"
    manifest_path = out_dir / "run_manifest.json"

    write_or_check_manifest(
        manifest_path,
        signature_payload,
        selected,
        runtimes,
        allow_config_change=args.allow_config_change,
    )

    tasks = build_tasks(selected, args.n_per_cell, args.seed, runtimes)
    latest = load_latest_rows(results_path)
    pending = [task for task in tasks if not (task.task_id in latest and is_task_complete(latest[task.task_id]))]

    completed_existing = len(tasks) - len(pending)
    if completed_existing:
        print(f"Resume: {completed_existing:,} completed model draws already present; {len(pending):,} tasks remain.")

    if not pending:
        totals = summarize_results(results_path, summary_path, args.n_per_cell)
        print(f"Nothing to run. Summary refreshed at {summary_path}.")
        print(json.dumps(totals, indent=2))
        return 0

    semaphores = {key: threading.Semaphore(args.per_provider_workers) for key in selected}
    max_workers = max(1, len(selected) * args.per_provider_workers)
    print(f"Executing with up to {args.per_provider_workers} concurrent request(s) per provider ({max_workers} total workers).")

    # Keep all writes on the main thread; each result row is flushed immediately.
    exists = results_path.exists() and results_path.stat().st_size > 0
    f = results_path.open("a", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
    if not exists:
        writer.writeheader()
        f.flush()

    done = 0
    ok = 0
    parse_errors = 0
    api_errors = 0
    try:
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm-exp") as executor:
            future_to_task = {
                executor.submit(
                    run_one_trial,
                    task,
                    runtimes[task.provider_key],
                    args.timeout,
                    args.max_retries,
                    temperature,
                    top_p,
                    semaphores[task.provider_key],
                    args.seed,
                ): task
                for task in pending
            }

            for future in as_completed(future_to_task):
                row = future.result()
                writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
                f.flush()
                done += 1
                if row["status"] == "ok":
                    ok += 1
                elif row["status"] == "parse_error":
                    parse_errors += 1
                else:
                    api_errors += 1

                if done == 1 or done % 50 == 0 or done == len(pending):
                    print(
                        f"Progress {done:,}/{len(pending):,} | ok={ok:,} | parse_error={parse_errors:,} | api_error={api_errors:,}",
                        flush=True,
                    )
    except KeyboardInterrupt:
        print("\nInterrupted. Completed rows have already been flushed; rerun the same command to resume.", file=sys.stderr)
        return 130
    finally:
        f.close()

    totals = summarize_results(results_path, summary_path, args.n_per_cell)
    print(f"\nFinished. Response-level data: {results_path}")
    print(f"Aggregate summary: {summary_path}")
    print(f"Run manifest: {manifest_path}")
    print(json.dumps(totals, indent=2))

    if totals["api_errors"] or totals["parse_errors"]:
        print(
            "WARNING: the latest dataset contains API or parse errors. API-error task IDs are retried automatically on the next run; "
            "parse errors are preserved as model draws and are not silently re-sampled.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
