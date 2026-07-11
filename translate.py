"""
translate.py — LLM-based subtitle translation

Translates subtitle/text files to a target language using a local GGUF model via llama-server (llama.cpp).  
Designed for both standalone use and integration
from ``main.py`` / ``batch_pipeline.py`` via the ``-translate`` flag.

Can translate SRT (preserving timecodes) and plain TXT files.

Standalone usage:
  python translate.py -i output/input.txt                  # → output/input_CN.txt + .srt
  python translate.py -i output/input.srt                  # preserves SRT timecodes
  python translate.py -i output/input.txt -o output/my_trans.srt
  python translate.py -i output/input.txt -tgt-lang Japanese -tgt-lang-code JP
  python translate.py -i output/input.srt -tgt-lang French -tgt-lang-code FR

Integration (add ``-translate`` to existing commands):
  python main.py -i video.mp4 -translate
  python batch_pipeline.py -translate

══════════════════════════════════════════════════════════════════════════════
  USER CONFIGURATION  —  edit ``translate_config.json`` in the project root to customise behaviour.  
  The variables below are defaults that the JSON file overrides at startup.
══════════════════════════════════════════════════════════════════════════════
"""

import json
import os
import re
import sys
import subprocess
import time
import argparse
from pathlib import Path

import requests

# ── Project root — used early so load_config() can find the JSON file ────────
# When packaged (PyInstaller), user-facing paths live alongside the .exe;
# bundled internal files (tools/llama/) are inside _internal/ or MEIPASS.
if getattr(sys, "frozen", False):
    BASE_DIR = Path(sys.executable).parent.resolve()
    _internal = BASE_DIR / "_internal"
    if _internal.is_dir():
        _BUNDLE_DIR = _internal
    elif hasattr(sys, "_MEIPASS"):
        _BUNDLE_DIR = Path(sys._MEIPASS)
    else:
        _BUNDLE_DIR = BASE_DIR
else:
    BASE_DIR = Path(__file__).parent.resolve()
    _BUNDLE_DIR = BASE_DIR

# ══════════════════════════════════════════════════════════════════════════════
#  ENV LOADER  —  loads ``.env`` from the project root (if present) into
#  ``os.environ`` so API keys can stay out of version control.
# ══════════════════════════════════════════════════════════════════════════════

_ENV_PATH = BASE_DIR / ".env"


def _load_env():
    """Load KEY=VALUE pairs from .env into os.environ (setdefault)."""
    if not _ENV_PATH.exists():
        return
    with open(_ENV_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)


_load_env()

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIG LOADER  —  loads ``translate_config.json`` from the project root
# ══════════════════════════════════════════════════════════════════════════════

_CONFIG_PATH = BASE_DIR / "translate_config.json"


def _load_config():
    """Load configuration from JSON file, falling back to built-in defaults."""
    defaults = {
        "target_lang": "Chinese",
        "target_lang_code": "CN",
        "source_lang": "English",
        "add_punctuation": False,
        "allow_flexible_word_order": False,
        "allow_simplify_wording": False,
        "number_mode": "auto",
        "space_between_cjk_and_latin": False,
        "glossary": [],
        "custom_system_prompt": None,
        "cache_prompt": False,
        "openai_api_key": "",
        "anthropic_api_key": "",
        "api_base_url": "",
    }
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                overrides = json.load(f)
            # Merge: JSON values override defaults; extra JSON keys are ignored
            result = {**defaults, **overrides}
            # Strip the comment key if present (""_"" is a JSON comment hint)
            result.pop("_", None)
            result.pop("_comment", None)
            # Fall back to env vars for API keys (loaded from .env if present)
            if not result.get("openai_api_key"):
                result["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
            if not result.get("anthropic_api_key"):
                result["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
            print(f"  [Config] Loaded: {_CONFIG_PATH}")
            return result
        except Exception as e:
            print(f"  [WARN] Failed to parse {_CONFIG_PATH}: {e}")
            print(f"         Falling back to defaults.")
    else:
        print(f"  [Config] {_CONFIG_PATH} not found, using defaults")
    return defaults


CFG = _load_config()

# Convenience aliases (read-only after startup — edit the JSON file instead)
TARGET_LANG = CFG["target_lang"]
TARGET_LANG_CODE = CFG["target_lang_code"]
SOURCE_LANG = CFG["source_lang"]
ADD_PUNCTUATION = CFG["add_punctuation"]
ALLOW_FLEXIBLE_WORD_ORDER = CFG["allow_flexible_word_order"]
ALLOW_SIMPLIFY_WORDING = CFG["allow_simplify_wording"]
NUMBER_MODE = CFG["number_mode"]
SPACE_BETWEEN_CJK_AND_LATIN = CFG["space_between_cjk_and_latin"]
GLOSSARY = CFG["glossary"]
CUSTOM_SYSTEM_PROMPT = CFG["custom_system_prompt"]
CACHE_PROMPT = CFG["cache_prompt"]
OPENAI_API_KEY = CFG["openai_api_key"]


# ── Debug flags ───────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = CFG["anthropic_api_key"]
API_BASE_URL = CFG["api_base_url"]


# ── Paths & constants ──────────────────────────────────────────────────────
LLAMA_SERVER = _BUNDLE_DIR / "tools" / "llama" / "llama-server.exe"
NO_PROXY = {"http": None, "https": None}

_MODEL_REGISTRY = {
    "phi4":      "phi-4-Q4_K_M.gguf",
}
_MODEL_LAYERS = {
    "phi4":      40,
}

# ── Translation backends ────────────────────────────────────────────────────
# Default base URLs and model names for each supported API backend.
BACKEND_DEFAULTS = {
    "deepseek":  {"base_url": "https://api.deepseek.com",
                  "default_model": "deepseek-v4-flash"},
    "openai":    {"base_url": "https://api.openai.com",
                  "default_model": "gpt-5.6-terra"},
    "qwen":      {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                  "default_model": "qwen3.5-plus"},
    "gemini":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                  "default_model": "gemini-3.5-flash"},
    "ollama":    {"base_url": "http://localhost:11434/v1",
                  "default_model": "llama4"},
    "anthropic": {"base_url": "https://api.anthropic.com",
                  "default_model": "claude-opus-4-8"},
}

_SRT_TIME_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})"
)


# ══════════════════════════════════════════════════════════════════════════════
#  GPU layer auto-detection  (mirrors tools/llm_pipeline.py)
# ══════════════════════════════════════════════════════════════════════════════

def auto_gpu_layers(model="phi4"):
    """Estimate how many model layers fit in GPU VRAM."""
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    try:
        props = torch.cuda.get_device_properties(0)
        total_vram = props.total_memory / 1e9
        model_file = _MODEL_REGISTRY.get(model)
        if not model_file:
            return 0
        model_path = BASE_DIR / "models" / model_file
        if not model_path.exists():
            return 0
        model_size = os.path.getsize(model_path) / 1e9
        n_layers = _MODEL_LAYERS.get(model, 0)
        if n_layers == 0:
            return 0
        available = total_vram - 1.5
        if available <= 0:
            return 0
        per_layer = model_size / n_layers * 1.1
        return max(0, min(n_layers, int(available / per_layer)))
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════════════════
#  Prompt building
# ══════════════════════════════════════════════════════════════════════════════

def _build_system_prompt(tgt_lang, src_lang, add_punct, glossary,
                         flexible_word_order=None,
                         simplify_wording=None,
                         number_mode=None):
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


def _fmt_prompt(model, system, user):
    """Wrap system + user in the Phi-4 chat template."""
    return (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{user}\n<|end|>\n"
        f"<|assistant|>\n"
    )


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
    """Insert a space between CJK characters and Latin letters/digits."""
    if not SPACE_BETWEEN_CJK_AND_LATIN:
        return text
    text = re.sub(r"([一-鿿])([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])([一-鿿])", r"\1 \2", text)
    text = re.sub(r"([一-鿿])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([一-鿿])", r"\1 \2", text)
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
#  Translation backends
# ══════════════════════════════════════════════════════════════════════════════


class TranslatorLocal:
    """Local GGUF model inference via llama-server subprocess."""

    def __init__(self, model="phi4", port=18081, gpu_layers=None,
                 num_threads=12, context_size=4096):
        self.model = model
        model_file = _MODEL_REGISTRY.get(model)
        if not model_file:
            raise ValueError(
                f"Unknown model: {model!r}. "
                f"Available: {list(_MODEL_REGISTRY)}"
            )
        self.model_path = str(BASE_DIR / "models" / model_file)
        self.server_path = str(LLAMA_SERVER)
        self.port = port
        self.server_url = f"http://127.0.0.1:{port}"
        self.gpu_layers = auto_gpu_layers(model) if gpu_layers is None else gpu_layers
        self.num_threads = num_threads
        self.context_size = context_size
        self.server_proc = None
        self._stderr_log = None

    def start(self, wait_up_to=120):
        """Start llama-server and wait for it to respond."""
        if self.server_proc:
            return True

        print(f"  [Translate] Starting {self.model.upper()} (llama-server)...",
              end=" ", flush=True)

        cmd = [
            self.server_path, "-m", self.model_path,
            "--port", str(self.port),
            "-t", str(self.num_threads),
            "-c", str(self.context_size),
            "--cont-batching",
            "-ngl", str(self.gpu_layers),
        ]
        self._stderr_log = open("llama_stderr.log", "w", encoding="utf-8")
        self.server_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=self._stderr_log,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )

        for i in range(wait_up_to):
            try:
                r = requests.get(f"{self.server_url}/health", timeout=2,
                                 proxies=NO_PROXY)
                if r.status_code == 200:
                    print(f"ok ({i + 1}s)")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        print("FAIL")
        return False

    def translate_batch(self, system, user, grammar=None):
        """Send a batch of text to the local LLM and return the raw response."""
        prompt = _fmt_prompt(self.model, system, user)
        n_pred = max(200, int(len(user) * 1.5))

        body = {
            "prompt": prompt,
            "n_predict": n_pred,
            "temperature": 0.1,
            "cache_prompt": CACHE_PROMPT,
        }
        if grammar:
            body["grammar"] = grammar

        for attempt in range(2):
            try:
                r = requests.post(
                    f"{self.server_url}/completion",
                    json=body,
                    timeout=120,
                    proxies=NO_PROXY,
                )
                if r.status_code == 200:
                    r.encoding = 'utf-8'
                    return r.json().get("content", "").strip()
                print(f"    LLM HTTP {r.status_code}: {r.text[:300]}", flush=True)
                if r.status_code == 500 and attempt == 0:
                    print("    Retrying...", flush=True)
                    continue
                break
            except Exception as e:
                print(f"    LLM error: {e}", flush=True)
                break

        # ── On persistent HTTP 500, salvage output from stderr ──────────
        # llama-server logs the model output to stderr as a WARNING before
        # returning the PEG-parser 500 error.  We can extract it from there.
        content = self._salvage_stderr_content()
        if content:
            print(f"    [Salvaged] {len(content)} chars from stderr", flush=True)
            return content

        # All attempts failed — dump server stderr for diagnosis
        if self._stderr_log:
            self._stderr_log.flush()
            try:
                with open(self._stderr_log.name, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                    tail = "".join(lines[-60:])
                    print(f"    ── llama-server stderr (tail 60) ──", flush=True)
                    for _l in tail.split("\n"):
                        if _l.strip():
                            print(f"    | {_l}", flush=True)
                    print(f"    ────────────────────────────────────", flush=True)
            except Exception as e:
                print(f"    (error reading stderr log: {e})", flush=True)
        return None

    def _salvage_stderr_content(self):
        """Try to extract the model output from PEG-parser warning in stderr.

        The PEG parser logs ``unparsed Content-only output: <text>`` to stderr
        *before* returning HTTP 500, so the content is always available there.
        """
        if not self._stderr_log:
            return None
        self._stderr_log.flush()
        try:
            with open(self._stderr_log.name, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
        except Exception:
            return None

        # Strip ANSI colour codes, then find the last PEG parser line
        clean = _strip_ansi(data)
        m = re.search(r"unparsed Content-only output:\s*(.+)", clean, re.DOTALL)
        if not m:
            return None
        content = m.group(1).strip()
        # Remove U+FFFD / invalid UTF-8 leftovers
        content = content.replace("�", "").strip()
        return content or None

    def stop(self):
        """Gracefully stop llama-server."""
        proc, self.server_proc = self.server_proc, None
        if proc is None:
            return
        pid = proc.pid
        print(f"  [Translate] Shutting down llama-server (PID {pid})...",
              end=" ", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=5)
            print("ok")
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=3)
                print("ok (kill)")
            except Exception:
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid)],
                    capture_output=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                print("ok (taskkill)")
        if self._stderr_log:
            self._stderr_log.close()
            self._stderr_log = None


class TranslatorOpenAI:
    """OpenAI-compatible API backend (DeepSeek, OpenAI, Qwen, Ollama, …)."""

    def __init__(self, backend, model, api_key, base_url_override=""):
        self.backend = backend
        self.model = model
        cfg = BACKEND_DEFAULTS[backend]
        self.base_url = (base_url_override or cfg["base_url"]).rstrip("/")
        self.api_key = api_key

    def translate_batch(self, system, user):
        """Send a batch via the chat completions endpoint."""
        max_tokens = max(200, int(len(user) * 1.5))
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.1,
            "max_tokens": max_tokens,
        }
        # DeepSeek V4 defaults to thinking mode, which eats the token budget
        # leaving content empty.  Explicitly disable it when using DeepSeek.
        if self.backend == "deepseek":
            body["thinking"] = {"type": "disabled"}

        try:
            r = requests.post(
                f"{self.base_url}/chat/completions",
                json=body,
                headers=headers,
                timeout=120,
                proxies=NO_PROXY,
            )
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"].strip()
            print(f"    API HTTP {r.status_code}: {r.text[:300]}", flush=True)
        except Exception as e:
            print(f"    API error: {e}", flush=True)
        return None

    def stop(self):
        """No-op for API backends — nothing to shut down."""
        pass


class TranslatorAnthropic:
    """Anthropic Claude API backend."""

    def __init__(self, model, api_key, base_url_override=""):
        self.model = model
        cfg = BACKEND_DEFAULTS["anthropic"]
        self.base_url = (base_url_override or cfg["base_url"]).rstrip("/")
        self.api_key = api_key

    def translate_batch(self, system, user):
        """Send a batch via the Anthropic Messages API."""
        max_tokens = max(200, int(len(user) * 1.5))

        try:
            r = requests.post(
                f"{self.base_url}/v1/messages",
                json={
                    "model": self.model,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": 0.1,
                    "max_tokens": max_tokens,
                    "thinking": {"type": "disabled"},
                },
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                timeout=120,
                proxies=NO_PROXY,
            )
            if r.status_code == 200:
                content_blocks = r.json()["content"]
                for block in content_blocks:
                    if block.get("type") == "text":
                        return block["text"].strip()
                print("    [WARN] No text block in response", flush=True)
                return ""
            print(f"    API HTTP {r.status_code}: {r.text[:200]}", flush=True)
        except Exception as e:
            print(f"    API error: {e}", flush=True)
        return None

    def stop(self):
        """No-op for API backends — nothing to shut down."""
        pass


def create_translator(backend, model=None, gpu_layers=None):
    """Factory: return a translator instance for the given backend."""
    if backend == "local":
        m = model or "phi4"
        return TranslatorLocal(model=m, gpu_layers=gpu_layers)

    if backend not in BACKEND_DEFAULTS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Available: local, {', '.join(BACKEND_DEFAULTS)}"
        )

    cfg = BACKEND_DEFAULTS[backend]
    m = model or cfg["default_model"]

    if backend == "anthropic":
        api_key = ANTHROPIC_API_KEY
        if not api_key:
            print("  [WARN] ANTHROPIC_API_KEY is not set in translate.py config")
        return TranslatorAnthropic(model=m, api_key=api_key,
                                   base_url_override=API_BASE_URL)

    if backend == "deepseek":
        # DeepSeek supports two API formats; auto-detect from base_url:
        #   "https://api.deepseek.com"           → OpenAI /chat/completions
        #   "https://api.deepseek.com/anthropic" → Anthropic Messages API
        api_key = OPENAI_API_KEY
        if not api_key:
            print("  [WARN] OPENAI_API_KEY is not set in translate.py config")
        base_url = API_BASE_URL or "https://api.deepseek.com"
        if "anthropic" in base_url.lower():
            return TranslatorAnthropic(model=m, api_key=api_key,
                                       base_url_override=base_url)
        return TranslatorOpenAI(backend=backend, model=m,
                                api_key=api_key,
                                base_url_override=base_url)

    # All other backends use the OpenAI-compatible format
    # Ollama doesn't need a key; others use OPENAI_API_KEY
    return TranslatorOpenAI(backend=backend, model=m,
                            api_key=OPENAI_API_KEY,
                            base_url_override=API_BASE_URL)


# ══════════════════════════════════════════════════════════════════════════════
#  Translation engine
# ══════════════════════════════════════════════════════════════════════════════

def translate_texts(texts, server, tgt_lang=None, src_lang=None,
                    add_punct=None, glossary=None, flexible_word_order=None,
                    simplify_wording=None, number_mode=None, mode="accurate",
                    online=False):
    """Translate a list of text strings via the LLM server.

    Args:
        texts: list of source-language strings, OR list of SRT block dicts
               (from ``parse_srt``).  When SRT blocks are passed, timecodes
               are automatically included in the prompt.
        server: started translator instance.
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
                                  number_mode)

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
        response = server.translate_batch(system, user)
        if response is None:
            print("    [WARN] LLM call failed, keeping original text")
            for j in range(actual_tr):
                results[tr_start + j] = text_lines[tr_start + j]
            ctx_start += ctx_step
            tr_start += tr_size
            ctx_step = later_step
            continue

        retry_delays = [5, 15]
        for attempt, delay in enumerate(retry_delays):
            if response.strip():
                break
            print(f"    [WARN] Empty response, retrying in {delay}s "
                  f"({attempt + 1}/{len(retry_delays)})...")
            time.sleep(delay)
            response = server.translate_batch(system, user)
            if response is None:
                break
        if not response or not response.strip():
            print(f"    [WARN] API returned empty response after retries, "
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
        for resp_line in response.split("\n"):
            resp_line = _strip_template_tokens(resp_line.strip())
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
                    fallback_lines.append(resp_line)

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
                            if online:
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

    # ── Post-process: fix overflow merges (online models only) ─────
    if online and overflow_records:
        results = _fix_overflow(results, overflow_records, server,
                                text_lines, tgt_lang, src_lang, add_punct)

    return results


def _fix_overflow(results, overflow_records, server, text_lines,
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

        response = server.translate_batch(system, user)
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
    parser.add_argument("-tgt-lang", default=None,
                        help=f"Target language name (default: {TARGET_LANG})")
    parser.add_argument("-tgt-lang-code", default=None,
                        help=f"Language code for filenames (default: {TARGET_LANG_CODE})")
    parser.add_argument("-src-lang", default=None,
                        help=f"Source language name (default: {SOURCE_LANG})")
    parser.add_argument("-backend", default="local",
                        choices=["local"] + list(BACKEND_DEFAULTS.keys()),
                        help=f"Translation backend: local, {', '.join(BACKEND_DEFAULTS)} "
                             f"(default: local)")
    parser.add_argument("-mode", default="accurate",
                        choices=["accurate", "flexible"],
                        help="Translation mode: accurate (sliding-window, "
                             "no timecodes) or flexible (larger windows, "
                             "timecode-aware, for online APIs). "
                             "(default: accurate)")
    parser.add_argument("-model", default=None,
                        help="Model name for the selected backend "
                             "(e.g. deepseek-v4-flash, gpt-5.6-terra, "
                             "gemini-3.5-flash, claude-opus-4-8; "
                             "default: per-backend default)")
    parser.add_argument("-gpu-layers", type=int, default=None,
                        help="GPU layers (local backend only; default: auto-detect)")
    parser.add_argument("-no-cache", action="store_true",
                        help="Disable prompt caching (ignored if not needed)")
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
        print(f"  python {Path(__file__).name} -i output/input.txt -tgt-lang Japanese -tgt-lang-code JP")
        print(f"  python {Path(__file__).name} -i output/input.srt")
        print(f"  python {Path(__file__).name} -i output/input.txt -o D:/output/lecture_CN.srt")
        print()
        print("For details: use -h or --help")
        print()
        sys.exit(1)

    # ── Resolve language ──────────────────────────────────────────────
    tgt_lang = args.tgt_lang or TARGET_LANG
    tgt_code = args.tgt_lang_code or TARGET_LANG_CODE
    src_lang = args.src_lang or SOURCE_LANG

    print("=" * 55)
    print(f"  Translate: {src_lang} → {tgt_lang} ({tgt_code})")
    print(f"  Mode:      {args.mode}")
    print("=" * 55)

    # ── Read input ────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERR] Input not found: {input_path}")
        sys.exit(1)

    texts, is_srt = read_input(str(input_path))
    blocks = texts if is_srt else None
    text_lines = [b["text"] for b in texts] if is_srt else texts

    if not text_lines:
        print("[ERR] No text found in input file")
        sys.exit(1)

    # ── Resolve output path ───────────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_dir = output_path.parent.resolve()
        stem = output_path.with_suffix("").stem
    else:
        # Default: output/<input_stem>_<LANG_CODE>.srt/.txt
        input_stem = input_path.stem
        # Strip any previous language suffix to avoid double suffixes
        for code in (tgt_code, tgt_code.lower()):
            if input_stem.endswith(f"_{code}"):
                input_stem = input_stem[:-(len(code) + 1)]
                break
        stem = f"{input_stem}_{tgt_code}"
        # Default to project-root output/ directory
        output_dir = BASE_DIR / "output"

    print(f"  Input:  {input_path}")
    print(f"  Output: {output_dir / stem}.[srt|txt]")
    print()

    # ── Resolve backend & start translator ─────────────────────────
    backend = args.backend
    print(f"  Backend: {backend}")

    if backend == "local":
        translator = create_translator("local", model=args.model or "phi4",
                                       gpu_layers=args.gpu_layers)
        if not translator.start():
            print("[ERR] Failed to start llama-server")
            sys.exit(1)
    else:
        translator = create_translator(backend, model=args.model)

    try:
        results = translate_texts(
            texts if is_srt else text_lines, translator,
            tgt_lang=tgt_lang, src_lang=src_lang,
            add_punct=ADD_PUNCTUATION, glossary=GLOSSARY,
            number_mode=NUMBER_MODE, mode=args.mode,
            online=(backend != "local"),
        )
        if results is None:
            print("[ERR] Translation failed")
            sys.exit(1)

        write_output(stem, output_dir, results,
                     is_srt=is_srt, blocks=blocks)

    finally:
        translator.stop()

    print()
    print("=" * 55)
    print("  [OK] Translation complete!")
    print("=" * 55)


if __name__ == "__main__":
    main()
