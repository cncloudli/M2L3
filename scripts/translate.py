"""
translate.py — LLM-based subtitle translation

Translates subtitle/text files to a target language using an LLM backend
(see :mod:`tools.llm_call`). Designed for both standalone use and integration
from ``main.py`` via the ``-translate`` flag.

Can translate SRT (preserving timecodes) and plain TXT files.

Standalone usage:
  python scripts/translate.py -i output/input.txt          # → output/input_CN.txt + .srt
  python scripts/translate.py -i output/input.srt          # preserves SRT timecodes
  python scripts/translate.py -i output/input.txt -o output/my_trans.srt
  python scripts/translate.py -i output/input.txt -tgt_lang Japanese -tgt_lang_code JP

══════════════════════════════════════════════════════════════════════════════
  USER CONFIGURATION  —  edit ``translate_config.json`` in the project root to customise behaviour.
══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import re
import sys
import time
import argparse
from pathlib import Path

# ── Project root — must be on sys.path before tools imports ──────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.llm_call import create_llm_call, load_api_config, BACKEND_DEFAULTS, strip_reasoning, get_min_tokens

# ── Base directories for config/data paths ──────────────────────────────────
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent.resolve()
    _BUNDLE_DIR = (BASE_DIR / "_internal") if (BASE_DIR / "_internal").is_dir() else BASE_DIR
else:
    BASE_DIR = Path(__file__).resolve().parent.parent  # scripts/ → project root
    _BUNDLE_DIR = BASE_DIR

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG LOADER  —  loads translation config from ``translate_config.json``
#  (API keys are handled by :func:`tools.llm_call.load_api_config`)
# ══════════════════════════════════════════════════════════════════════════════

_CONFIG_PATH = BASE_DIR / "translate_config.json"


def _load_translation_config():
    """Load translation-specific configuration from JSON file."""
    defaults = {
        "target_lang": "Chinese",
        "target_lang_code": "CN",
        "source_lang": "English",
        "add_punctuation": False,
        "allow_flexible_word_order": False,
        "allow_simplify_wording": False,
        "number_mode": "auto",
        "space_between_cjk_and_latin": "auto",
        "glossary": [],
        "custom_system_prompt": None,
        "cache_prompt": False,
        "drift_threshold": False,
    }
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            result = {**defaults, **overrides}
            result.pop("_", None)
            result.pop("_comment", None)
            print(f"  [Config] Loaded: {_CONFIG_PATH}")
            return result
        except Exception as e:
            print(f"  [WARN] Failed to parse {_CONFIG_PATH}: {e}")
    return dict(defaults)


CFG = _load_translation_config()

# Convenience aliases (read-only after startup)
TARGET_LANG = CFG["target_lang"]
TARGET_LANG_CODE = CFG["target_lang_code"]
SOURCE_LANG = CFG["source_lang"]

# ══════════════════════════════════════════════════════════════════════════════
#  Language code mapping  —  bidirectional lookup for validation & auto-fill
# ══════════════════════════════════════════════════════════════════════════════

_LANG_TO_CODE = {
    "chinese": "CN",
    "english": "EN",
    "japanese": "JP",
    "korean": "KR",
    "french": "FR",
    "german": "DE",
    "spanish": "ES",
    "italian": "IT",
    "portuguese": "PT",
    "russian": "RU",
    "arabic": "AR",
    "hindi": "HI",
    "thai": "TH",
    "vietnamese": "VN",
    "dutch": "NL",
    "polish": "PL",
    "turkish": "TR",
    "indonesian": "ID",
    "malay": "MS",
    "swedish": "SV",
    "norwegian": "NO",
    "danish": "DA",
    "finnish": "FI",
    "greek": "EL",
    "hebrew": "HE",
    "romanian": "RO",
    "czech": "CS",
    "hungarian": "HU",
    "ukrainian": "UK",
}

_CODE_TO_LANG = {v: k for k, v in _LANG_TO_CODE.items()}


def _resolve_language_params(tgt_lang=None, tgt_code=None, src_lang=None):
    """Resolve and validate language parameters with auto-completion.

    Priority: explicit argument (CLI / function parameter) > config file > hardcoded default.
    If only one of ``tgt_lang`` / ``tgt_code`` is given, the other is
    auto-completed from the built-in mapping table (``_LANG_TO_CODE`` /
    ``_CODE_TO_LANG``).
    If both are given, they are validated for consistency.
    If neither is given, config file defaults are used.

    Returns:
        ``(resolved_tgt_lang, resolved_tgt_code, resolved_src_lang)``.
    """
    resolved_src = src_lang or SOURCE_LANG

    # Determine what was explicitly provided
    tgt_from_cli = tgt_lang is not None
    code_from_cli = tgt_code is not None

    # Start with config values as baseline
    resolved_tgt = TARGET_LANG
    resolved_code = TARGET_LANG_CODE

    # Apply explicit overrides
    if tgt_from_cli:
        resolved_tgt = tgt_lang
    if code_from_cli:
        resolved_code = tgt_code

    # Validate / auto-complete the target language + code pair
    if tgt_from_cli and code_from_cli:
        _validate_lang_code_pair(resolved_tgt, resolved_code)
    elif tgt_from_cli and not code_from_cli:
        mapped = _LANG_TO_CODE.get(resolved_tgt.lower().strip())
        if mapped:
            resolved_code = mapped
            print(f"  [Lang] Auto-completed -tgt_lang_code → {resolved_code}")
    elif code_from_cli and not tgt_from_cli:
        mapped = _CODE_TO_LANG.get(resolved_code.upper().strip())
        if mapped:
            resolved_tgt = mapped
            print(f"  [Lang] Auto-completed -tgt_lang → {resolved_tgt}")
    # else: neither → keep config defaults

    return resolved_tgt, resolved_code, resolved_src


def _validate_lang_code_pair(tgt, code):
    """Exit with an error if *tgt* and *code* are inconsistent."""
    expected_code = _LANG_TO_CODE.get(tgt.lower().strip())
    expected_lang = _CODE_TO_LANG.get(code.upper().strip())
    if expected_code and expected_code != code.upper().strip():
        print(f"[ERR] -tgt_lang '{tgt}' maps to code '{expected_code}', "
              f"but -tgt_lang_code '{code}' was given. "
              f"These must be consistent.")
        sys.exit(1)
    if expected_lang and expected_lang != tgt.lower().strip():
        print(f"[ERR] -tgt_lang_code '{code}' maps to language "
              f"'{expected_lang.capitalize()}', but -tgt_lang '{tgt}' was "
              f"given. These must be consistent.")
        sys.exit(1)
ADD_PUNCTUATION = CFG["add_punctuation"]
ALLOW_FLEXIBLE_WORD_ORDER = CFG["allow_flexible_word_order"]
ALLOW_SIMPLIFY_WORDING = CFG["allow_simplify_wording"]
NUMBER_MODE = CFG["number_mode"]
SPACE_BETWEEN_CJK_AND_LATIN = CFG["space_between_cjk_and_latin"]
GLOSSARY = CFG["glossary"]
CUSTOM_SYSTEM_PROMPT = CFG["custom_system_prompt"]
CACHE_PROMPT = CFG["cache_prompt"]
DRIFT_THRESHOLD = CFG["drift_threshold"]

_SRT_TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt building
# ══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(tgt_lang, src_lang, add_punct, glossary,
                         flexible_word_order=None,
                         simplify_wording=None,
                         number_mode=None,
                         model=None):
    """Build the system (instruction) prompt from user configuration."""
    if CUSTOM_SYSTEM_PROMPT:
        return CUSTOM_SYSTEM_PROMPT

    flexible_word_order = (
        ALLOW_FLEXIBLE_WORD_ORDER if flexible_word_order is None
        else flexible_word_order
    )
    simplify_wording = (
        ALLOW_SIMPLIFY_WORDING if simplify_wording is None
        else simplify_wording
    )
    number_mode = (
        NUMBER_MODE if number_mode is None
        else number_mode
    )

    parts = [
        f"Translate each line of {src_lang} text below to {tgt_lang}."
    ]
    if add_punct:
        parts.append(
            "Add appropriate punctuation in the translation, "
            "including commas and sentence-ending punctuation (。？！). "
            "Every sentence should end with proper punctuation."
        )
    if flexible_word_order:
        parts.append(
            "IMPORTANT — Word reordering is permitted. "
            "You may adjust word order and redistribute content "
            "across adjacent lines within a batch to produce "
            "natural-sounding translations. "
            "You may also merge multiple short adjacent lines into "
            "a single translated sentence — use the range output "
            "format to indicate this."
        )
    else:
        parts.append(
            "CRITICAL — Translate each line independently. "
            "Do NOT rearrange or redistribute content across lines. "
            "However, if two short adjacent lines read more naturally "
            "as a single sentence in the target language, you may "
            "merge them — use the range output format to indicate "
            "which source lines were combined."
        )
    if simplify_wording:
        parts.append(
            "Condense colloquial or verbose expressions into concise, "
            "natural-sounding language — remove filler words, redundancies, "
            "and rambling constructions. CRITICAL: Preserve ALL substantive "
            "information. NEVER change the meaning or omit key details."
        )
    if glossary:
        terms = ", ".join(glossary)
        parts.append(
            "CRITICAL — Do NOT translate these terms. "
            "Preserve them EXACTLY as written in the source language: "
            f"{terms}. "
            "You must keep these words in their original form, even if you "
            "know their translation."
        )
    if number_mode and number_mode != "auto":
        if number_mode == "src_lang":
            parts.append(
                "CRITICAL — Number preservation rule: Preserve ALL numbers "
                "EXACTLY as they appear in the source text. Do not convert "
                "any numerals or number words between languages or formats. "
                "The output numbers must look identical to the source numbers."
            )
        elif number_mode == "digits":
            parts.append(
                "CRITICAL — Number format rule: ALL numbers MUST use Arabic "
                "digits (0‑9) only. Write \"100\" — NOT \"one hundred\" "
                "and NOT \"一百\". Write \"42\" — NOT \"forty-two\" and NOT "
                "\"四十二\". This applies to every number in the output "
                "without exception."
            )
        elif number_mode == "tgt_lang":
            parts.append(
                "CRITICAL — Number format rule: ALL numbers MUST use the "
                "target language's native numeric form. When translating to "
                "Chinese: write '一百' — NOT 100 and NOT 'one hundred'. "
                "Write '四十二' — NOT 42 and NOT 'forty-two'. This applies "
                "to every number without exception."
            )
    parts.append(
        'CRITICAL — Output format: Start each line with the source line '
        'number in brackets, followed by the translation.\n'
        'Examples:\n'
        '[1] Translation of source line 1.\n'
        '[2-3] Merged translation of source lines 2 and 3.\n'
        '[4] Translation of source line 4.\n'
        'If a single output covers source lines N through M, use [N-M] — '
        'this lets the system merge their timestamps.\n'
        "Output ONLY these bracketed lines. Do not include "
        "any explanations, notes, or meta-commentary — not a single extra word."
    )

    return "\n".join(parts)



def _build_window_user_prompt(ctx_lines, tr_start_idx, tr_end_idx,
                               show_timecodes=False):
    """Build a user prompt with context-only and translate sections.

    Args:
        ctx_lines: list of strings or SRT block dicts in the context window.
        tr_start_idx: 0-based index within *ctx_lines* to start translating.
        tr_end_idx: 0-based exclusive end index.
        show_timecodes: if True and items are SRT block dicts, include
                        ``[start --> end]`` in the prompt.

    The translate section numbers its lines **from 1**, so the LLM's output
    format ``[1] text`` maps directly to the first translate line.
    Context-only lines are prefixed with a clear marker so the LLM knows not
    to translate them.
    """
    def _format_line(item, idx=None, context_label=None):
        if isinstance(item, dict):
            tc = f" [{item['start']} --> {item['end']}]" if show_timecodes else ""
            if idx:
                return f"{idx}{tc} {item['text']}"
            label = f"({context_label})" if context_label else " "
            return f"{label}{tc} {item['text']}"
        if idx:
            return f"{idx} {item}"
        label = f"({context_label})" if context_label else " "
        return f"{label} {item}"

    parts = []

    context_before = ctx_lines[:tr_start_idx]
    translate_part = ctx_lines[tr_start_idx:tr_end_idx]
    context_after = ctx_lines[tr_end_idx:]

    if context_before:
        parts.append("CONTEXT (reference only, do not translate):")
        for item in context_before:
            parts.append(_format_line(item, context_label="BEFORE"))
        parts.append("")

    parts.append("TRANSLATE THESE LINES (output [1], [2], ...):")
    for i, item in enumerate(translate_part, 1):
        parts.append(_format_line(item, idx=i))
    parts.append("")

    if context_after:
        parts.append("CONTEXT (reference only, do not translate):")
        for item in context_after:
            parts.append(_format_line(item, context_label="AFTER"))

    return "\n".join(parts)


_ANSI_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def _strip_ansi(text):
    """Strip ANSI escape sequences from *text*."""
    return _ANSI_RE.sub("", text)


# ══════════════════════════════════════════════════════════════════════════════
#  SRT parsing / reconstruction
# ══════════════════════════════════════════════════════════════════════════════

def parse_srt(text):
    """Parse SRT text into a list of ``{start, end, text}`` dicts."""
    blocks = []
    lines = text.strip().split("\n")
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if not line.isdigit():
            i += 1
            continue
        i += 1
        if i >= len(lines):
            break
        m = _SRT_TIME_RE.match(lines[i].strip())
        if not m:
            i += 1
            continue
        start = m.group(1).replace(".", ",")
        end = m.group(2).replace(".", ",")
        i += 1
        text_lines = []
        while i < len(lines) and lines[i].strip():
            text_lines.append(lines[i].strip())
            i += 1
        text = " ".join(text_lines)
        blocks.append({"start": start, "end": end, "text": text})
        i += 1
    return blocks


def rebuild_srt(blocks, translated_texts):
    """Reconstruct SRT string, merging timecodes for merged translations.

    When ``translated_texts[i]`` is ``None``, the LLM merged that line's
    content into an adjacent translation.  The block's timecode is absorbed
    into the nearest surviving block so timing stays correct.
    """
    out = []
    i = 0
    seq = 1  # sequential subtitle number
    while i < len(blocks):
        # Skip blocks that were absorbed by a neighbor (None translation)
        if i < len(translated_texts) and translated_texts[i] is None:
            i += 1
            continue

        # Keep original text for blocks beyond translated range (safety)
        text = translated_texts[i] if i < len(translated_texts) else blocks[i]["text"]
        start = blocks[i]["start"]

        # Look ahead: absorb trailing Nones into this block's timecode
        end = blocks[i]["end"]
        j = i + 1
        while j < len(blocks):
            if j < len(translated_texts) and translated_texts[j] is not None:
                break  # reached a block with its own translation
            end = blocks[j]["end"]
            j += 1

        out.append(str(seq))
        seq += 1
        out.append(f"{start} --> {end}")
        out.append(text)
        out.append("")
        i = j

    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
#  Post-processing
# ══════════════════════════════════════════════════════════════════════════════

def _add_cjk_latin_spacing(text):
    """Insert, remove, or keep spaces between CJK characters and Latin letters/digits.

    - True / "true":   insert a space between CJK and Latin characters/digits.
    - False / "false": strip any spaces between CJK and Latin characters/digits
                       (overrides the LLM's own spacing habits).
    - "auto":          return the LLM output unchanged.
    """
    val = SPACE_BETWEEN_CJK_AND_LATIN
    if isinstance(val, str) and val.lower() == "auto":
        return text
    if val is True or (isinstance(val, str) and val.lower() == "true"):
        text = re.sub(r"([一-鿿])([a-zA-Z])", r"\1 \2", text)
        text = re.sub(r"([a-zA-Z])([一-鿿])", r"\1 \2", text)
        text = re.sub(r"([一-鿿])(\d)", r"\1 \2", text)
        text = re.sub(r"(\d)([一-鿿])", r"\1 \2", text)
    else:
        text = re.sub(r"([一-鿿]) ([a-zA-Z])", r"\1\2", text)
        text = re.sub(r"([a-zA-Z]) ([一-鿿])", r"\1\2", text)
        text = re.sub(r"([一-鿿]) (\d)", r"\1\2", text)
        text = re.sub(r"(\d) ([一-鿿])", r"\1\2", text)
    return text


def _strip_template_tokens(text):
    """Remove LLM template tokens like <|end|>, <|assistant|>, etc."""
    return re.sub(r"<\|.*?\|>", "", text).strip()


def _strip_end_punct(text):
    """Remove sentence-ending punctuation from the end of the string.

    Strips 。？！.!? from the very end.
    Keeps commas (，,), colons, ellipsis-like sequences, and all internal punctuation intact.
    """
    return text.rstrip("。？！.!?")



# Languages that use full-width punctuation (，。！？ etc.)
_FULLWIDTH_PUNCT_LANGUAGES = frozenset({
    "chinese", "mandarin", "cantonese",
    "japanese",
    "korean",
})


def _needs_fullwidth_punct(lang):
    """Check if the target language uses full-width punctuation."""
    return lang.strip().lower() in _FULLWIDTH_PUNCT_LANGUAGES


def _normalize_commas(text, tgt_lang):
    """Convert half-width commas to full-width (，) for CJK languages."""
    if not _needs_fullwidth_punct(tgt_lang):
        return text
    return text.replace(",", "，")


def _normalize_quotes(text, tgt_lang):
    """Convert double quotes based on the quoted content's language.

    For CJK target languages: if the quoted content contains CJK characters,
    use fullwidth quotes (""); otherwise keep halfwidth ("").
    For non-CJK languages: always keep halfwidth.
    """
    if not _needs_fullwidth_punct(tgt_lang):
        return text
    return re.sub(
        r'"([^"]*)"',
        lambda m: '“' + m.group(1) + '”'
        if re.search(r'[一-鿿]', m.group(1))
        else '"' + m.group(1) + '"',
        text,
    )


def post_process(text, tgt_lang="Chinese", add_punct=True):
    """Run all post-processing steps on a translated string."""
    text = _strip_template_tokens(text)
    if not add_punct:
        text = _strip_end_punct(text)
    text = _normalize_commas(text, tgt_lang)
    text = _normalize_quotes(text, tgt_lang)
    return _add_cjk_latin_spacing(text)


# ══════════════════════════════════════════════════════════════════════════════
#  Translation engine
# ══════════════════════════════════════════════════════════════════════════════

def translate(texts,
              transl_backend="local", transl_model=None,
              api_key="", base_url="", gpu_layers=None,
              tgt_lang=None, src_lang=None,
              add_punct=None, glossary=None, flexible_word_order=None,
              simplify_wording=None, number_mode=None, mode="accurate"):
    """Translate subtitle/text lines via LLM.

    Creates an LLM backend internally, starts/stops it, and runs the
    sliding-window translation engine.

    Args:
        texts: list of source-language strings, or list of SRT block dicts
               from ``parse_srt``.
        transl_backend: LLM backend name (local, deepseek, openai, …).
        transl_model: model name (None → per-backend default).
        api_key: API key for online backends.
        base_url: base URL override for online backends.
        gpu_layers: GPU offload layers (local backend only; None = auto).
        tgt_lang: target language name.
        src_lang: source language name.
        add_punct: add punctuation in translation.
        glossary: terms to keep untranslated.
        flexible_word_order: allow cross-line rephrasing.
        simplify_wording: condense colloquial text.
        number_mode: number handling mode.
        mode: "accurate" or "flexible".

    Returns:
        list of translated strings (same order as *texts*), or ``None``
        on critical failure.
    """
    llm_call = create_llm_call(
        backend=transl_backend, model=transl_model,
        api_key=api_key, base_url=base_url, gpu_layers=gpu_layers,
    )
    if transl_backend == "local":
        if not llm_call.start():
            print("[ERR] LLM server failed to start", flush=True)
            return None

    try:
        return _translate_impl(
            texts, llm_call,
            tgt_lang=tgt_lang, src_lang=src_lang,
            add_punct=add_punct, glossary=glossary,
            flexible_word_order=flexible_word_order,
            simplify_wording=simplify_wording,
            number_mode=number_mode, mode=mode,
            online=(transl_backend != "local"),
        )
    finally:
        llm_call.stop()


def _translate_impl(texts, llm_call, tgt_lang=None, src_lang=None,
                    add_punct=None, glossary=None, flexible_word_order=None,
                    simplify_wording=None, number_mode=None, mode="accurate",
                    online=False):
    """Translate a list of text strings via the LLM server.

    Args:
        texts: list of source-language strings, OR list of SRT block dicts
               (from ``parse_srt``).  When SRT blocks are passed, timecodes
               are automatically included in the prompt.
        llm_call: started :class:`tools.llm_call.LLMCall` instance.
        tgt_lang: target language name (default: ``TARGET_LANG``).
        src_lang: source language name (default: ``SOURCE_LANG``).
        add_punct: punctuation flag (default: ``ADD_PUNCTUATION``).
        glossary: terms to keep untranslated (default: ``GLOSSARY``).
        flexible_word_order: allow cross-line rephrasing (default: ``ALLOW_FLEXIBLE_WORD_ORDER``).
        simplify_wording: condense colloquial/verbose text (default: ``ALLOW_SIMPLIFY_WORDING``).
        number_mode: number handling mode (default: ``NUMBER_MODE``).
        mode: "accurate" or "flexible" (default: "accurate").

    Returns:
        List of translated strings (same order as *texts*), or ``None`` on
        critical failure.
    """
    tgt_lang = tgt_lang or TARGET_LANG
    src_lang = src_lang or SOURCE_LANG
    add_punct = ADD_PUNCTUATION if add_punct is None else add_punct
    glossary = GLOSSARY if glossary is None else glossary
    simplify_wording = (ALLOW_SIMPLIFY_WORDING if simplify_wording is None
                        else simplify_wording)
    number_mode = NUMBER_MODE if number_mode is None else number_mode

    # Auto-detect SRT blocks (dicts with start/end/text) vs plain strings
    if texts and isinstance(texts[0], dict) and "start" in texts[0]:
        blocks = texts
        text_lines = [b["text"] for b in blocks]
    else:
        blocks = None
        text_lines = texts

    results = [None] * len(text_lines)

    # ── Sliding-window parameters (mode-dependent) ────────────────────
    if mode == "flexible":
        tr_size, ctx_size, first_step, later_step = 4, 8, 2, 4
        show_tc = True
        mode_label = "flexible"
    else:
        tr_size, ctx_size, first_step, later_step = 2, 4, 1, 2
        show_tc = False
        mode_label = "accurate"

    # ── Mode-based parameter guard ─────────────────────────────────────
    # allow_flexible_word_order and allow_simplify_wording only have
    # meaning in flexible mode; force them off in accurate mode even if
    # the user mistakenly enabled them in translate_config.json.
    if mode != "flexible":
        flexible_word_order = False
        simplify_wording = False

    system = _build_system_prompt(tgt_lang, src_lang, add_punct, glossary,
                                  flexible_word_order, simplify_wording,
                                  number_mode,
                                  model=getattr(llm_call, 'model', ''))

    ctx_start = 0
    tr_start = 0
    ctx_step = first_step
    overflow_records = []

    while tr_start < len(text_lines):
        tr_end = min(tr_start + tr_size, len(text_lines))
        actual_tr = tr_end - tr_start

        # Ensure context window covers the translate window
        ctx_end = max(ctx_start + ctx_size, tr_end)
        ctx_end = min(ctx_end, len(text_lines))
        if ctx_start > tr_start:
            ctx_start = tr_start

        # Slice the window from source
        ctx_slice = (blocks[ctx_start:ctx_end] if blocks
                     else text_lines[ctx_start:ctx_end])
        tr_local_start = tr_start - ctx_start
        tr_local_end = tr_end - ctx_start

        # Build user prompt with context/translate sections
        user = _build_window_user_prompt(
            ctx_slice, tr_local_start, tr_local_end,
            show_timecodes=show_tc,
        )

        print(f"  [Translate] ({mode_label}) Lines "
              f"{tr_start + 1}–{tr_end}/{len(text_lines)}", flush=True)

        # ── API call ──────────────────────────────────────────────────
        response = llm_call.chat(system, user,
                                  max_tokens=get_min_tokens(
                                      getattr(llm_call, 'model', ''),
                                      len(user)),
                                  temperature=0.1, cache_prompt=CACHE_PROMPT)
        if response:
            response = strip_reasoning(response)

        retry_delays = [5, 15, 30]
        for attempt, delay in enumerate(retry_delays):
            if response and response.strip():
                break
            reason = "LLM call failed" if response is None else "Empty response"
            print(f"    [WARN] {reason}, retrying in {delay}s "
                  f"({attempt + 1}/{len(retry_delays)})...")
            time.sleep(delay)
            response = llm_call.chat(system, user,
                                   max_tokens=get_min_tokens(
                                       getattr(llm_call, 'model', ''),
                                       len(user)),
                                   temperature=0.1, cache_prompt=CACHE_PROMPT)
            if response:
                response = strip_reasoning(response)
        if not response or not response.strip():
            print(f"    [WARN] LLM call failed after retries, "
                  f"keeping original text")
            for j in range(actual_tr):
                results[tr_start + j] = text_lines[tr_start + j]
            ctx_start += ctx_step
            tr_start += tr_size
            ctx_step = later_step
            continue

        if tr_start + tr_size < len(text_lines):
            time.sleep(0.8)

        # Strip cached prompt history
        response = re.split(r"<\|", response)[0].strip()

        # ── Parse LLM response ───────────────────────────────────────
        # Each output line is expected to be formatted as:
        #   [N] text  or  [N-M] text
        # N and M refer to the 1-based numbering in the "TRANSLATE" section.
        parsed_items = []
        fallback_lines = []

        # Defensive: strip any leading bracketed timecodes that some models
        # sometimes reproduce from the user prompt.
        _LEADING_TC_RE = re.compile(
            r"^\[\d{2}:\d{2}:\d{2}[，,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[，,.]\d{3}\]\s*"
        )

        for resp_line in response.split("\n"):
            resp_line = _strip_template_tokens(resp_line.strip())
            resp_line = _LEADING_TC_RE.sub("", resp_line).strip()
            if not resp_line:
                continue
            m = re.match(
                r"\[(\d+)(?:\s*-\s*(\d+))?\]\s*"
                r"(?:\[([\d:,]+\s*-->\s*[\d:,]+)\]\s*)?"
                r"(.+)",
                resp_line,
            )
            if m:
                n = int(m.group(1)) - 1       # 0‑based in translate section
                m_n = int(m.group(2)) - 1 if m.group(2) else n
                llm_tc = m.group(3)
                text = m.group(4).strip()
                parsed_items.append((n, m_n, llm_tc, text))
            else:
                # Fallback: old N: format
                m2 = re.match(r"(?:Line\s*)?(\d+)\s*[:：]\s*(.+)",
                              resp_line, re.IGNORECASE)
                if m2:
                    n = int(m2.group(1)) - 1
                    text = m2.group(2).strip()
                    if 0 <= n < actual_tr:
                        parsed_items.append((n, n, None, text))
                else:
                    # Fallback: bare N prefix — some models drop
                    # brackets from "[N] text", producing "N text".
                    m3 = re.match(r"(\d+)\s+(.+)", resp_line)
                    if m3:
                        n = int(m3.group(1)) - 1
                        text = m3.group(2).strip()
                        if 0 <= n < actual_tr:
                            # Strip any leading timecode the model reproduced
                            text = _LEADING_TC_RE.sub("", text).strip()
                            parsed_items.append((n, n, None, text))
                    else:
                        fallback_lines.append(resp_line)

        # ── CJK verification: retry if parsed output is all English ─────
        # Some models sometimes echo source text verbatim instead of
        # translating — either with brackets (parsed_items) or without
        # (fallback_lines). We detect this by checking whether ANY output
        # line contains CJK characters; if none do, the model failed to
        # translate and we retry.
        _CJK_CHECK_RE = re.compile(r"[一-鿿㐀-䶿　-〿＀-￯]")
        _all_output_texts = (
            [t for _, _, _, t in parsed_items]
            if parsed_items else fallback_lines
        )
        if _all_output_texts and not any(
            _CJK_CHECK_RE.search(t) for t in _all_output_texts
        ):
            print(f"    [WARN] Window {tr_start+1}–{tr_end}: LLM output "
                  f"contains no CJK (all English) — retrying…", flush=True)
            for retry_idx in range(2):
                retry_system = (
                    "CRITICAL — You failed in your previous attempt. "
                    "You output the English source text instead of "
                    "translating to Chinese.\n\n"
                    "This is your retry. You MUST now:\n"
                    "1. Translate EVERY source line to Chinese.\n"
                    "2. Output ONLY Chinese text in every [N] line.\n"
                    "3. If unsure about a term, embed the English word "
                    "inside the Chinese sentence — do NOT leave the "
                    "entire line untranslated.\n\n"
                    "Output ONLY [N] lines with Chinese translations. "
                    "No English text in the [N] lines."
                )
                retry_resp = llm_call.chat(
                    retry_system, user,
                    max_tokens=get_min_tokens(
                        getattr(llm_call, 'model', ''), len(user)),
                    temperature=0, cache_prompt=False,
                )
                if retry_resp:
                    retry_resp = strip_reasoning(retry_resp)
                    retry_resp = re.split(r"<\|", retry_resp)[0].strip()

                # Re-parse the retry response
                new_parsed = []
                new_fallback = []
                for resp_line in (retry_resp or "").split("\n"):
                    resp_line = _strip_template_tokens(resp_line.strip())
                    resp_line = _LEADING_TC_RE.sub("", resp_line).strip()
                    if not resp_line:
                        continue
                    m = re.match(
                        r"\[(\d+)(?:\s*-\s*(\d+))?\]\s*"
                        r"(?:\[([\d:,]+\s*-->\s*[\d:,]+)\]\s*)?"
                        r"(.+)", resp_line,
                    )
                    if m:
                        n = int(m.group(1)) - 1
                        m_n = int(m.group(2)) - 1 if m.group(2) else n
                        text = m.group(4).strip()
                        new_parsed.append((n, m_n, m.group(3), text))
                    else:
                        m2 = re.match(
                            r"(?:Line\s*)?(\d+)\s*[:：]\s*(.+)",
                            resp_line, re.IGNORECASE)
                        if m2:
                            n = int(m2.group(1)) - 1
                            text = m2.group(2).strip()
                            if 0 <= n < actual_tr:
                                new_parsed.append((n, n, None, text))
                        else:
                            m3 = re.match(r"(\d+)\s+(.+)", resp_line)
                            if m3:
                                n = int(m3.group(1)) - 1
                                text = m3.group(2).strip()
                                if 0 <= n < actual_tr:
                                    text = _LEADING_TC_RE.sub("", text).strip()
                                    new_parsed.append((n, n, None, text))
                            else:
                                new_fallback.append(resp_line)

                if new_parsed and any(
                    _CJK_CHECK_RE.search(text) for _, _, _, text in new_parsed
                ):
                    print(f"    [OK] Retry produced CJK content — using it")
                    parsed_items = new_parsed
                    fallback_lines = new_fallback
                    break
                elif not new_parsed and new_fallback:
                    # LLM still ignored format — check fallback CJK
                    if any(_CJK_CHECK_RE.search(t) for t in new_fallback):
                        print(f"    [INFO] Retry fallback has CJK — using it")
                        fallback_lines = new_fallback
                        parsed_items = []
                        break
                    else:
                        print(f"    [WARN] Retry {retry_idx + 1} fallback "
                              f"also has no CJK, retrying…")
                        # Continue loop for another retry
                else:
                    print(f"    [WARN] Retry {retry_idx + 1} also produced "
                          f"no CJK, retrying…" if retry_idx == 0
                          else f"    [WARN] Retry {retry_idx + 1} also "
                          f"produced no CJK, keeping original")
            else:
                # All retries exhausted — keep original (English) output
                print(f"    [WARN] CJK retry exhausted — keeping "
                      f"English output for window {tr_start+1}–{tr_end}")

        # ── Resolve positions ────────────────────────────────────────
        translated_local = {}

        if not parsed_items:
            # LLM ignored format — treat as one-per-line fallback
            for j, rl in enumerate(fallback_lines):
                if j < actual_tr:
                    translated_local[j] = rl
        else:
            # Trust sequence numbers — [N] maps to position N, [N-M] merges
            for n, m_n, _tc, text in parsed_items:
                if 0 <= n < actual_tr:
                    merge_end = min(m_n, actual_tr - 1)
                    translated_local[n] = text
                    for j in range(n + 1, merge_end + 1):
                        translated_local[j] = None

                    # Detect overflow: LLM merged content beyond translate window
                    if m_n >= actual_tr:
                        _overflowed = tr_start + actual_tr
                        if _overflowed < len(text_lines):
                            _merged = tr_start + n
                            overflow_records.append((_merged, _overflowed))

        # ── Store results ────────────────────────────────────────────
        for j in range(actual_tr):
            global_idx = tr_start + j
            if j in translated_local:
                if translated_local[j] is not None:
                    results[global_idx] = post_process(
                        translated_local[j], tgt_lang, add_punct=add_punct,
                    )
                else:
                    results[global_idx] = None
            else:
                results[global_idx] = None

        # ── Print preview lines ─────────────────────────────────────
        for j in range(actual_tr):
            idx = tr_start + j
            orig = text_lines[idx]
            trans = results[idx]
            if trans is None:
                print(f"    {idx + 1}: {orig[:60]}{'…' if len(orig) > 60 else ''}")
                print(f"       → # merged into adjacent subtitle #")
            elif orig != trans:
                short_orig = orig[:60] + ("…" if len(orig) > 60 else "")
                short_trans = trans[:60] + ("…" if len(trans) > 60 else "")
                print(f"    {idx + 1}: {short_orig}")
                print(f"       → {short_trans}")

        # ── Advance windows ──────────────────────────────────────────
        ctx_start += ctx_step
        tr_start += tr_size
        ctx_step = later_step

    # ── Post-process: fix overflow merges ──────────────────────────
    if overflow_records:
        results = _fix_overflow(results, overflow_records, llm_call,
                                text_lines, tgt_lang, src_lang, add_punct)

    # ── Post-process: detect & fix content drift ──────────────────
    if DRIFT_THRESHOLD is not False:
        drift_indices = _detect_drift_pairs(results, text_lines,
                                             threshold=DRIFT_THRESHOLD)
        if drift_indices:
            print(f"\n  [Drift] {len(drift_indices)} line(s) flagged, "
                  f"re-translating individually...")
            results = _fix_drift_lines(results, drift_indices, llm_call,
                                        text_lines, blocks,
                                        tgt_lang, src_lang, add_punct, glossary)
            print(f"  [Drift] Done — {len(drift_indices)} line(s) processed\n")

    # ── Post-process: fix individual English lines (CJK check) ────
    # Catches lines that passed the per-window CJK check because their
    # paired window-mate had CJK, but are themselves in English.
    _CJK_CHECK_RE = re.compile(r"[一-鿿㐀-䶿　-〿＀-￯]")
    _cjk_miss = [
        i for i, r in enumerate(results)
        if r is not None and i < len(text_lines)
        and not _CJK_CHECK_RE.search(r)
        and (results[i] == text_lines[i] or len(r.split()) >= 3)  # LLM-call-failed / original-preserved lines: no length limit
        and any(c.isalpha() for c in r)  # has actual text
    ]
    if _cjk_miss:
        print(f"\n  [CJK Fix] {len(_cjk_miss)} line(s) lack CJK, "
              f"re-translating individually...")
        _cjk_system = _build_system_prompt(
            tgt_lang, src_lang, add_punct, glossary,
            flexible_word_order=False, simplify_wording=False,
            number_mode=NUMBER_MODE,
            model=getattr(llm_call, 'model', ''),
        )
        # Determine character descriptor for the target language
        _lang_lower = tgt_lang.lower()
        if "japanese" in _lang_lower:
            _char_desc = "Japanese characters"
        elif "korean" in _lang_lower:
            _char_desc = "Korean characters"
        else:
            _char_desc = "Chinese characters"
        _cjk_system += (
            "\n\nCRITICAL: Translate ONLY the single line below. "
            "Output format: [1] <translation>. "
            f"The translation MUST contain {_char_desc}."
        )
        for idx in _cjk_miss:
            item = blocks[idx] if blocks else text_lines[idx]
            _user = _build_window_user_prompt(
                [item], 0, 1, show_timecodes=False)
            _resp = llm_call.chat(
                _cjk_system, _user,
                max_tokens=get_min_tokens(
                    getattr(llm_call, 'model', ''), len(_user)),
                temperature=0, cache_prompt=False,
            )
            if _resp:
                _resp = strip_reasoning(_resp)
            _m = re.match(r"\[1(?:\s*-\s*1)?\]\s*(.+)", (_resp or "").strip())
            if _m:
                _new = post_process(_m.group(1).strip(), tgt_lang,
                                    add_punct=add_punct)
                if _new and _CJK_CHECK_RE.search(_new):
                    print(f"    [CJK Fix] Line {idx + 1}: fixed")
                    print(f"      Old: {results[idx][:80]}")
                    print(f"      New: {_new[:80]}")
                    results[idx] = _new
                else:
                    print(f"    [CJK Fix] Line {idx + 1}: new output "
                          f"also no CJK, keeping original")
            else:
                print(f"    [CJK Fix] Line {idx + 1}: could not parse "
                      f"response, keeping original")
        print(f"  [CJK Fix] Done — {len(_cjk_miss)} line(s) processed\n")

    return results


def _fix_overflow(results, overflow_records, llm_call, text_lines,
                  tgt_lang, src_lang, add_punct):
    """Post-process: re-split overflow merges via LLM.

    When the LLM merges content from beyond the translate window (e.g. '[2-3]'
    when only lines 1-2 should be translated), the merged text at
    ``results[merged_idx]`` contains content belonging to
    ``results[overflowed_idx]``.  This function sends both to the LLM for a
    clean re-split so no content is duplicated.
    """
    print(f"\n  [Post-process] Fixing {len(overflow_records)} overflow merge(s)...")
    results = list(results)

    for merged_idx, overflowed_idx in overflow_records:
        if (merged_idx >= len(results) or overflowed_idx >= len(results)
                or results[merged_idx] is None or results[overflowed_idx] is None):
            continue

        src_a = text_lines[merged_idx]
        src_b = text_lines[overflowed_idx]
        trans_a = results[merged_idx]
        trans_b = results[overflowed_idx]

        system = (
            "You are a subtitle editor. Re-organise two adjacent subtitle "
            "translations that overlap in content. Remove any duplicate or "
            "overlapping content so each line contains only its own meaning."
        )
        user = (
            f"Two consecutive subtitle translations overlap in content. "
            f"The first accidentally includes content from the second line.\n\n"
            f"Source [{merged_idx + 1}]: {src_a}\n"
            f"Translation [{merged_idx + 1}]: {trans_a}\n\n"
            f"Source [{overflowed_idx + 1}]: {src_b}\n"
            f"Translation [{overflowed_idx + 1}]: {trans_b}\n\n"
            f"Re-split them so each translation covers only its own source line. "
            f"Output format:\n"
            f"[{merged_idx + 1}] <revised translation>\n"
            f"[{overflowed_idx + 1}] <revised translation>"
        )

        response = llm_call.chat(system, user,
                                  max_tokens=get_min_tokens(
                                      getattr(llm_call, 'model', ''),
                                      len(user)),
                                  temperature=0.1, cache_prompt=CACHE_PROMPT)
        if response:
            response = strip_reasoning(response)
        if not response or not response.strip():
            print(f"    [WARN] Re-split failed for lines "
                  f"{merged_idx + 1}–{overflowed_idx + 1}, keeping original")
            continue

        # Parse response
        parsed = {}
        for resp_line in response.split("\n"):
            resp_line = _strip_template_tokens(resp_line.strip())
            m = re.match(r"\[(\d+)\]\s*(.+)", resp_line)
            if m:
                parsed[int(m.group(1)) - 1] = m.group(2).strip()

        if merged_idx in parsed and overflowed_idx in parsed:
            results[merged_idx] = post_process(
                parsed[merged_idx], tgt_lang, add_punct=add_punct,
            )
            results[overflowed_idx] = post_process(
                parsed[overflowed_idx], tgt_lang, add_punct=add_punct,
            )
            print(f"    [Fix] Lines {merged_idx + 1} & {overflowed_idx + 1}: "
                  f"re-split OK")
        else:
            print(f"    [WARN] Could not parse re-split response for lines "
                  f"{merged_idx + 1}–{overflowed_idx + 1}")

    return results


# ── Content drift detection & fix ──────────────────────────────────────────

def _char_sim(a, b):
    """Jaccard character-set similarity between two strings.

    Defined as ``len(set(a) & set(b)) / len(set(a) | set(b))``.
    Returns 1.0 when both strings are empty.
    """
    if not a or not b:
        return 0.0
    set_a = set(a.strip())
    set_b = set(b.strip())
    if not set_a and not set_b:
        return 1.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 0.0


def _detect_drift_pairs(results, text_lines, threshold=0.16):
    """Detect adjacent translation pairs where content may have drifted.

    When two adjacent translations share more characters than their source
    counterparts (gap = trans_sim - src_sim > *threshold*), the LLM may
    have copied content from one line into the other.

    Returns a sorted list of 0‑based line indices flagged for re-translation.
    """
    flagged = set()
    for i in range(len(results) - 1):
        if results[i] is None or results[i + 1] is None:
            continue
        trans_sim = _char_sim(results[i], results[i + 1])
        src_sim = _char_sim(text_lines[i], text_lines[i + 1])
        gap = trans_sim - src_sim
        if gap > threshold:
            flagged.add(i)
            flagged.add(i + 1)
            print(f"    [Drift] Lines {i + 1}–{i + 2}: "
                  f"trans_sim={trans_sim:.3f} src_sim={src_sim:.3f} "
                  f"gap={gap:.3f}")
    return sorted(flagged)


def _fix_drift_lines(results, drift_indices, llm_call, text_lines, blocks,
                     tgt_lang, src_lang, add_punct, glossary):
    """Re-translate flagged lines individually to eliminate content drift.

    Each flagged line is sent to the LLM as a single-line request (no context,
    no adjacent lines).  If the new translation differs from the current one
    it replaces the old value; otherwise the original is kept.
    """
    results = list(results)
    system = _build_system_prompt(
        tgt_lang, src_lang, add_punct, glossary,
        flexible_word_order=False, simplify_wording=False,
        number_mode=NUMBER_MODE,
        model=getattr(llm_call, 'model', ''),
    )
    system += (
        "\n\nCRITICAL: Translate ONLY the single line provided below. "
        "Do NOT combine it with any other text. "
        "Output format: [1] <translation>"
    )

    for idx in drift_indices:
        item = blocks[idx] if blocks else text_lines[idx]
        user = _build_window_user_prompt([item], 0, 1, show_timecodes=False)

        response = llm_call.chat(
            system, user,
            max_tokens=get_min_tokens(
                getattr(llm_call, 'model', ''), len(user)),
            temperature=0.1, cache_prompt=False,
        )
        if response:
            response = strip_reasoning(response)
        if not response or not response.strip():
            continue

        # Parse [1] or [1-1] format
        m = re.match(r"\[1(?:\s*-\s*1)?\]\s*(.+)", response.strip())
        if not m:
            continue

        new_trans = post_process(m.group(1).strip(), tgt_lang,
                                 add_punct=add_punct)

        if new_trans and new_trans != results[idx]:
            print(f"    [Drift Fix] Line {idx + 1}: replaced")
            print(f"      Old: {results[idx][:80]}")
            print(f"      New: {new_trans[:80]}")
            results[idx] = new_trans
        else:
            print(f"    [Drift Fix] Line {idx + 1}: unchanged "
                  f"(re-translation matched original or failed)")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  File I/O
# ══════════════════════════════════════════════════════════════════════════════

def read_input(filepath):
    """Read an SRT or TXT file.

    Tries UTF-8-SIG first; if that fails (common on Windows with legacy
    system-encoding files), falls back to the OS locale encoding.

    Returns:
        (data, is_srt): for SRT, *data* is a list of block dicts and
        ``is_srt=True``; for TXT, *data* is a list of line strings and
        ``is_srt=False``.
    """
    try:
        with open(filepath, "r", encoding="utf-8-sig") as f:
            content = f.read()
    except UnicodeDecodeError:
        import locale
        _fallback = locale.getpreferredencoding()
        with open(filepath, "r", encoding=_fallback) as f:
            content = f.read()

    if _SRT_TIME_RE.search(content):
        blocks = parse_srt(content)
        print(f"  [Translate] Detected SRT: {len(blocks)} blocks")
        return blocks, True

    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
    print(f"  [Translate] Detected TXT: {len(lines)} lines")
    return lines, False


def write_output(stem, output_dir, texts, is_srt, blocks=None):
    """Write translated SRT and/or TXT files.

    Args:
        stem: output filename stem (without extension).
        output_dir: output directory.
        texts: translated text list.
        is_srt: whether the input was SRT.
        blocks: original SRT block dicts (only if ``is_srt``).

    Returns:
        ``(srt_path, txt_path)`` — either entry may be ``None``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    txt_path = output_dir / f"{stem}.txt"

    if is_srt and blocks:
        srt_content = rebuild_srt(blocks, texts)
        srt_path = output_dir / f"{stem}.srt"
        with open(srt_path, "w", encoding="utf-8-sig") as f:
            f.write(srt_content)
        print(f"  [Translate] SRT saved: {srt_path}")
        # TXT gets the same content as SRT (with timestamps)
        with open(txt_path, "w", encoding="utf-8-sig") as f:
            f.write(srt_content)
        print(f"  [Translate] TXT saved: {txt_path}")
        return str(srt_path), str(txt_path)

    # Plain TXT — replace merged (None) entries with empty lines
    with open(txt_path, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(t or "" for t in texts))
    print(f"  [Translate] TXT saved: {txt_path}")
    return str(txt_path), None


# ══════════════════════════════════════════════════════════════════════════════
#  High-level convenience
# ══════════════════════════════════════════════════════════════════════════════


def translate_file(input_path,
                   output_stem=None,
                   output_dir=None,
                   transl_backend="local",
                   transl_model=None,
                   api_key="",
                   base_url="",
                   gpu_layers=None,
                   tgt_lang=None,
                   tgt_lang_code=None,
                   src_lang=None,
                   add_punct=None,
                   glossary=None,
                   flexible_word_order=None,
                   simplify_wording=None,
                   number_mode=None,
                   mode="accurate"):
    """Read a subtitle/text file, translate it, and write the output.

    High-level convenience that wraps :func:`read_input`, :func:`translate`,
    and :func:`write_output` into a single call — the caller only needs to
    pass a file path.

    Args:
        input_path: Path to input file (.srt or .txt).
        output_stem: Output filename stem (without extension).
            Auto-generated from input filename if not given.
        output_dir: Output directory. Defaults to ``output/`` under project
            root.
        tgt_lang_code: Language code used in the auto-generated output stem
            (default: ``TARGET_LANG_CODE`` module constant).
        transl_backend, transl_model, …: Passed through to :func:`translate`.

    Returns:
        ``(srt_path, txt_path)`` — either may be ``None`` if the format
        doesn't produce that file type. Both ``None`` on critical failure.
    """
    # ── Resolve language parameters (CLI > config > defaults) ──────────
    tgt_lang, tgt_lang_code, src_lang = _resolve_language_params(
        tgt_lang, tgt_lang_code, src_lang,
    )

    # ── Read input ─────────────────────────────────────────────────────
    texts_or_blocks, is_srt = read_input(input_path)

    if is_srt:
        text_lines = [b["text"] for b in texts_or_blocks]
    else:
        text_lines = texts_or_blocks

    if not text_lines:
        print(f"[ERR] No text found in {input_path}")
        return None, None

    # ── Resolve output path ────────────────────────────────────────────
    lang_code = tgt_lang_code
    p = Path(input_path)
    if output_stem is None:
        stem = p.stem
        # Strip any previous language suffix to avoid double suffixes
        for code in (lang_code, lang_code.lower()):
            if stem.endswith(f"_{code}"):
                stem = stem[:-(len(code) + 1)]
                break
        stem = f"{stem}_{lang_code}"
    else:
        stem = output_stem

    if output_dir is None:
        output_dir = str(BASE_DIR / "output")

    # ── Translate ──────────────────────────────────────────────────────
    result = translate(
        texts_or_blocks,
        transl_backend=transl_backend,
        transl_model=transl_model,
        api_key=api_key,
        base_url=base_url,
        gpu_layers=gpu_layers,
        tgt_lang=tgt_lang,
        src_lang=src_lang,
        add_punct=add_punct,
        glossary=glossary,
        flexible_word_order=flexible_word_order,
        simplify_wording=simplify_wording,
        number_mode=number_mode,
        mode=mode,
    )
    if result is None:
        return None, None

    # ── Write output ───────────────────────────────────────────────────
    return write_output(stem, output_dir, result,
                        is_srt=is_srt,
                        blocks=texts_or_blocks if is_srt else None)


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    parser = argparse.ArgumentParser(
        description="Translate subtitle / text files using an LLM",
    )
    parser.add_argument("-i", "--input", default=None,
                        help="Input file (.srt or .txt)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path "
                             "(default: output/<input_stem>_<lang_code>.srt/.txt)")
    parser.add_argument("-tgt_lang", default=None,
                        help=f"Target language name (default: {TARGET_LANG})")
    parser.add_argument("-tgt_lang_code", default=None,
                        help=f"Language code for filenames (default: {TARGET_LANG_CODE})")
    parser.add_argument("-src_lang", default=None,
                        help=f"Source language name (default: {SOURCE_LANG})")
    parser.add_argument("-transl_backend", default="local",
                        choices=["local"] + list(BACKEND_DEFAULTS.keys()),
                        help=f"Translation backend: local, {', '.join(BACKEND_DEFAULTS)} "
                             f"(default: local)")
    parser.add_argument("-mode", default="accurate",
                        choices=["accurate", "flexible"],
                        help="Translation mode: accurate (sliding-window, "
                             "no timecodes) or flexible (larger windows, "
                             "timecode-aware, for online APIs). "
                             "(default: accurate)")
    parser.add_argument("-transl_model", default=None,
                        help="Model name for the selected backend "
                             "(e.g. deepseek-v4-flash, gpt-5.6-terra, "
                             "gemini-3.5-flash, claude-opus-4-8; "
                             "default: per-backend default)")
    parser.add_argument("-local_model", default=None,
                        help="Local model name (e.g. phi4, qwen3.5-9b, "
                             "ministral-3-8b; default: phi4). Applied when "
                             "-transl_backend is 'local' and no explicit "
                             "-transl_model is given.")
    parser.add_argument("-gpu-layers", type=int, default=None,
                        help="GPU layers (local backend only; default: auto-detect)")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Check input argument ──────────────────────────────────────────
    if not args.input:
        print()
        print(f"  python {Path(__file__).name} -i <input> [options]")
        print()
        print("Examples:")
        print(f"  python {Path(__file__).name} -i output/input.txt")
        print(f"  python {Path(__file__).name} -i output/input.txt "
              "-tgt_lang Japanese -tgt_lang_code JP")
        print(f"  python {Path(__file__).name} -i output/input.srt")
        print(f"  python {Path(__file__).name} -i output/input.txt "
              "-o D:/output/lecture_CN.srt")
        print()
        print("For details: use -h or --help")
        print()
        sys.exit(1)

    # ── Resolve language ──────────────────────────────────────────────
    tgt_lang, tgt_code, src_lang = _resolve_language_params(
        args.tgt_lang, args.tgt_lang_code, args.src_lang,
    )

    print("=" * 55)
    print(f"  Translate: {src_lang} → {tgt_lang} ({tgt_code})")
    print(f"  Mode:      {args.mode}")
    print("=" * 55)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERR] Input not found: {input_path}")
        sys.exit(1)

    # ── Resolve output path ───────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_dir = str(output_path.parent.resolve())
        stem = output_path.with_suffix("").name
    else:
        output_dir = None
        stem = None

    print(f"  Input:  {input_path}")
    print()

    # ── Resolve backend ───────────────────────────────────────────────
    backend = args.transl_backend
    model = args.transl_model
    if backend == "local" and model is None:
        model = args.local_model
    print(f"  Backend: {backend}  Model: {model or 'default'}")

    _api_cfg = load_api_config()
    _api_key = _api_cfg.get("openai_api_key", "")
    _api_base = _api_cfg.get("api_base_url", "")
    if backend == "anthropic":
        _api_key = _api_cfg.get("anthropic_api_key", "")

    srt_result, txt_result = translate_file(
        str(input_path),
        output_stem=stem,
        output_dir=output_dir,
        transl_backend=backend,
        transl_model=model,
        api_key=_api_key,
        base_url=_api_base,
        gpu_layers=args.gpu_layers,
        tgt_lang=tgt_lang,
        tgt_lang_code=tgt_code,
        src_lang=src_lang,
        add_punct=ADD_PUNCTUATION,
        glossary=GLOSSARY,
        number_mode=NUMBER_MODE,
        mode=args.mode,
    )
    if srt_result is None and txt_result is None:
        print("[ERR] Translation failed")
        sys.exit(1)

    print()
    print("=" * 55)
    print("  [OK] Translation complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()
