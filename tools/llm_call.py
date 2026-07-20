"""
tools/llm_call.py — Unified LLM calling layer

Provides a common ``chat(system, user, …)`` interface for all LLM backends
used by both :mod:`tools.segment` and :mod:`translate`.

Backends
--------
- ``LLMCallLocal`` — local GGUF model via llama-server (``/completion``, Phi-4 template)
- ``LLMCallOpenAI`` — OpenAI-compatible API (``/chat/completions``, for DeepSeek / OpenAI / Qwen / Ollama …)
- ``LLMCallAnthropic`` — Anthropic Messages API (``/v1/messages``)

Usage
-----
    call = create_llm_call("local", model="phi4", gpu_layers=None)
    call.start()
    reply = call.chat("system prompt", "user text", max_tokens=512)
    call.stop()
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests


# ══════════════════════════════════════════════════════════════════════════════
#  PyInstaller-aware paths
# ══════════════════════════════════════════════════════════════════════════════

if getattr(sys, "frozen", False):
    _APP_DIR = str(Path(sys.executable).parent.resolve())
    _internal = Path(sys.executable).parent / "_internal"
    if _internal.is_dir():
        _BUNDLE_DIR = str(_internal)
    elif hasattr(sys, "_MEIPASS"):
        _BUNDLE_DIR = sys._MEIPASS
    else:
        _BUNDLE_DIR = _APP_DIR
else:
    _BASE = Path(__file__).resolve().parent.parent  # tools/.. → project root
    _APP_DIR = str(_BASE)
    _BUNDLE_DIR = str(_BASE)


NO_PROXY = {"http": None, "https": None}


# ══════════════════════════════════════════════════════════════════════════════
#  Model registry  (used by auto_gpu_layers)
# ══════════════════════════════════════════════════════════════════════════════

_MODEL_REGISTRY = {
    # ── Microsoft Phi series (MIT) ──────────────────────────────────────
    "phi4":              "phi-4-Q4_K_M.gguf",              # 14B, 40 layers (default)
    # ── Qwen3.5 series (Apache 2.0, Feb 2026) ─────────────────────────
    "qwen3.5-9b":        "Qwen3.5-9B-Q8_0.gguf",           # 9B,  32 layers
    # ── Mistral Ministral 3 series (Apache 2.0) ────────────────────────
    "ministral-3-8b":              "Ministral-3-8B-Instruct-2512-Q8_0.gguf",           # 8B, 34 layers
    "ministral-3-14b-instruct":    "Ministral-3-14B-Instruct-2512-Q5_K_M.gguf",         # 14B, 40 layers
}

_MODEL_LAYERS = {
    "phi4":              40,
    "qwen3.5-9b":        32,
"ministral-3-8b":    34,
    "ministral-3-14b-instruct": 40,
}


# ══════════════════════════════════════════════════════════════════════════════
#  Backend defaults
# ══════════════════════════════════════════════════════════════════════════════

BACKEND_DEFAULTS = {
    "deepseek":  {"base_url": "https://api.deepseek.com",
                   "default_model": "deepseek-v4-flash"},
    "openai":    {"base_url": "https://api.openai.com",
                   "default_model": "gpt-5.6-terra"},
    "qwen":      {"base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                   "default_model": "qwen3.7-max"},
    "gemini":    {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                   "default_model": "gemini-3.5-flash"},
    "anthropic": {"base_url": "https://api.anthropic.com",
                   "default_model": "claude-opus-4-8"},
}


# ══════════════════════════════════════════════════════════════════════════════
#  Shared utilities
# ══════════════════════════════════════════════════════════════════════════════


# ── Model-specific chat templates ──────────────────────────────────────────

_MODEL_TEMPLATES = {
    "phi4":             "phi4",
    "qwen3.5-9b":       "qwen",
"ministral-3-8b":   "ministral",
    "ministral-3-14b-instruct": "ministral",
}

# Models that output reasoning/thinking before the actual answer.
# These need a larger token budget so the thinking preamble doesn't
# crowd out the real output, and benefit from ``strip_reasoning()``.
_REASONING_MODELS = frozenset({
    "qwen3.5-9b",
})


def is_reasoning_model(model):
    """Return ``True`` if *model* is known to output reasoning preamble."""
    return model in _REASONING_MODELS


def get_min_tokens(model, user_len):
    """Sensible ``max_tokens`` for *model* given user message length.

    Reasoning models get a 512-token floor so thinking preamble doesn't
    consume the entire budget.  Non-reasoning models use the standard 200.
    """
    floor = 512 if is_reasoning_model(model) else 200
    return max(floor, int(user_len * 1.5))


def _fmt_prompt(system, user, model="phi4"):
    """Wrap *system* and *user* messages in the correct chat template.

    Dispatches to the model-specific template registered in ``_MODEL_TEMPLATES``.
    Falls back to the Phi-4 template for unknown models.
    """
    tmpl = _MODEL_TEMPLATES.get(model, "phi4")

    # ── Qwen ChatML template ────────────────────────────────────────────
    if tmpl == "qwen":
        return (
            "<|im_start|>system\n"
            f"{system}\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    # ── Ministral-3-Instruct template (used by Ministral-3-8B & -14B-2512) ─
    if tmpl == "ministral":
        return (
            f"[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]\n"
            f"[INST]{user}[/INST]"
        )

    # ── Phi-4 template (also default fallback) ─────────────────────────
    return (
        f"<|system|>\n{system}\n<|end|>\n"
        f"<|user|>\n{user}\n<|end|>\n"
        f"<|assistant|>\n"
    )


def clean_llm_output(text):
    """Strip trailing meta-commentary (Note:, Explanation:, etc.) from LLM output."""
    for marker in (r"\bNote\s*:", r"\bExplanation\s*:", r"\bExample\s*:", r"\bDisclaimer\s*:"):
        m = re.split(marker, text, flags=re.IGNORECASE)
        text = m[0].strip()
    text = re.sub(r"\n\[.*?\].*", "", text).strip()
    return text


def strip_reasoning(text):
    """Strip thinking/reasoning preamble from reasoning-model output.

    Handles:
      - ``<think>...</think>`` / ``<reason>...</reason>`` XML-style tags
      - Leading English prose before the first structured ``[N]`` line
        (the expected output format from translation and segmentation prompts)

    Returns the cleaned text, or the original if no preamble is detected.
    """
    if not text:
        return text

    # Remove XML-style thinking tags (greedy, matching newlines)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reason>.*?</reason>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE)

    # If text after tag removal is empty, it was all thinking
    text = text.strip()
    if not text:
        return text

    # ── Check for preamble before the first structured output line ──────
    # The expected format starts with [N] ... for translation or [line_N] for segmentation.
    # If there is text before the first such line, it's likely a reasoning preamble.
    m = re.search(r"^\[", text, re.MULTILINE)
    if m and m.start() > 0:
        preamble = text[:m.start()].strip()
        body = text[m.start():]
        # Only strip if preamble doesn't itself contain structured output
        if preamble and not re.search(r"\[.*?\]", preamble):
            text = body

    return text.strip()


def auto_gpu_layers(model="phi4"):
    """Estimate how many model layers fit in GPU VRAM.

    Uses ``nvidia-smi`` to query actual free VRAM at call time
    (accounting for fragmentation left by prior pipeline stages),
    then computes the maximum layer count that fits with a
    1.2 GB buffer for CUDA context + KV cache.

    Falls back to the old ``total_vram - 1.5`` calculation
    when ``nvidia-smi`` is unavailable.
    Returns 0 when CUDA is unavailable or the model is unknown.
    """
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0

    n_layers = _MODEL_LAYERS.get(model, 0)
    if n_layers == 0:
        return 0
    model_file = _MODEL_REGISTRY.get(model)
    if not model_file:
        return 0
    model_path = os.path.join(_APP_DIR, "models", model_file)
    if not os.path.exists(model_path):
        return 0

    try:
        # ── Query actual free VRAM via nvidia-smi ──────────────
        # This accounts for memory still held by the CUDA driver
        # after empty_cache() and gives the real available pool.
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        free_mib = int(out.stdout.strip().splitlines()[0].strip())
        available = free_mib / 1024.0 - 1.2  # GB, minus 1.2 GB buffer
    except Exception:
        # Fallback: compute from total VRAM
        try:
            props = torch.cuda.get_device_properties(0)
            available = props.total_memory / 1e9 - 1.5
        except Exception:
            return 0

    model_size = os.path.getsize(model_path) / 1e9
    if available <= 0 or model_size <= 0:
        return 0
    per_layer = model_size / n_layers * 1.1  # 10 % overhead
    return max(0, min(n_layers, int(available / per_layer)))


# ══════════════════════════════════════════════════════════════════════════════
#  API config loader  (reads translate_config.json + .env)
# ══════════════════════════════════════════════════════════════════════════════


def load_api_config():
    """Load API keys and base-url override from ``translate_config.json`` and ``.env``.

    Returns:
        dict with keys ``openai_api_key``, ``anthropic_api_key``, ``api_base_url``.
    """
    # Try loading .env first so its values take precedence over JSON defaults
    env_path = Path(_APP_DIR) / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("\"'")
                if key:
                    os.environ.setdefault(key, value)

    result = {
        "openai_api_key": "",
        "anthropic_api_key": "",
        "api_base_url": "",
    }

    config_path = Path(_APP_DIR) / "translate_config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            result["openai_api_key"] = cfg.get("openai_api_key", "")
            result["anthropic_api_key"] = cfg.get("anthropic_api_key", "")
            result["api_base_url"] = cfg.get("api_base_url", "")
        except Exception:
            pass

    # Fall back to environment variables if config values are empty
    if not result["openai_api_key"]:
        result["openai_api_key"] = os.environ.get("OPENAI_API_KEY", "")
    if not result["anthropic_api_key"]:
        result["anthropic_api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    if not result["api_base_url"]:
        result["api_base_url"] = os.environ.get("API_BASE_URL", "")

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Base class
# ══════════════════════════════════════════════════════════════════════════════


class LLMCall:
    """Abstract base for LLM backends.

    Subclasses must implement :meth:`chat`.
    """

    def start(self):
        """Start the backend (no-op for stateless API backends)."""
        return True

    def stop(self):
        """Shut down the backend (no-op for stateless API backends)."""
        pass

    def chat(self, system, user, max_tokens=None, temperature=0, cache_prompt=True):
        """Send a prompt to the LLM and return the response text.

        Args:
            system: system message (instructions / role).
            user: user message (the text to process).
            max_tokens: maximum tokens in the response (None → auto-compute).
            temperature: sampling temperature (0 = deterministic).
            cache_prompt: enable prompt caching (local backend only; ignored by API backends).

        Returns:
            Response string, or ``None`` on failure.
        """
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
#  Local backend  (llama-server)
# ══════════════════════════════════════════════════════════════════════════════


_LLAMASERVER = os.path.join(_BUNDLE_DIR, "tools", "llama", "llama-server.exe")


class LLMCallLocal(LLMCall):
    """Local GGUF model inference via llama-server subprocess.

    Wraps the llama-server lifecycle (start / stop) and sends requests
    to the ``/completion`` endpoint using the model-specific chat template
    registered in ``_MODEL_TEMPLATES``.
    """

    def __init__(self, model="phi4", model_path=None, server_path=None,
                 port=8088, gpu_layers=None, num_threads=12, context_size=4096):
        self.model = model
        if model_path is not None:
            self.model_path = model_path
        elif model in _MODEL_REGISTRY:
            self.model_path = os.path.join(_APP_DIR, "models",
                                           _MODEL_REGISTRY[model])
        else:
            self.model_path = os.path.join(_APP_DIR, "models", "phi-4-Q4_K_M.gguf")
        self.server_path = server_path or _LLAMASERVER
        self.port = port
        self.server_url = f"http://127.0.0.1:{port}"
        self.gpu_layers = gpu_layers  # None = auto-detect on start()
        self.num_threads = num_threads
        self.context_size = context_size
        self.server_proc = None
        self._server_pid = None  # saved for atexit / fallback kill
        self._cleanup_registered = False

    # ── Port cleanup ──────────────────────────────────────────────────────

    @staticmethod
    def _pid_exists(pid):
        """Check if a process with the given PID is alive on the system.

        Verifies via ``tasklist`` to avoid acting on stale netstat entries.
        """
        try:
            r = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                capture_output=True, text=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            # tasklist returns "INFO: No tasks are running..." when PID not found
            out = r.stdout.strip()
            return "No tasks" not in out and str(pid) in out.split()
        except Exception:
            return False

    @staticmethod
    def _find_pid_on_port(port):
        """Return the PID of the process listening on *port*, or ``None``."""
        try:
            out = subprocess.run(
                ["netstat", "-ano", "-p", "tcp"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            for line in out.stdout.splitlines():
                if f":{port}" in line and "LISTENING" in line:
                    parts = line.strip().split()
                    if parts:
                        pid = parts[-1]
                        if pid.isdigit():
                            return int(pid)
        except Exception:
            pass
        return None

    @staticmethod
    def _kill_pid(pid, log_prefix=""):
        """Kill a process by PID via ``taskkill /F``.

        Returns ``True`` if the kill was accepted, ``False`` on failure.
        """
        try:
            r = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if r.returncode != 0:
                err = r.stderr.decode("oem", errors="replace").strip()
                print(f"{log_prefix}taskkill /F /PID {pid} failed "
                      f"(rc={r.returncode}): {err}", flush=True)
                return False
            return True
        except Exception as e:
            print(f"{log_prefix}taskkill /F /PID {pid} raised: {e}", flush=True)
            return False

    def _free_port(self, max_retries=3):
        """Kill any existing process on ``self.port`` so we can start fresh.

        Retries up to *max_retries* times, verifying the port is actually free.
        Skip kills for stale netstat entries (PID no longer exists).
        """
        for attempt in range(max_retries):
            pid = self._find_pid_on_port(self.port)
            if pid is None:
                return True
            if not self._pid_exists(pid):
                # Stale netstat entry — process already dead, port is free
                time.sleep(0.5)
                continue
            print(f"    Port {self.port} in use by PID {pid}, killing "
                  f"(attempt {attempt + 1}/{max_retries})…", flush=True)
            ok = self._kill_pid(pid, log_prefix="    ")
            time.sleep(1)
            if not ok:
                # taskkill failed — try with /T to kill the process tree
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=5,
                        creationflags=subprocess.CREATE_NO_WINDOW,
                    )
                except Exception:
                    pass
                time.sleep(1)
        # Final check
        if self._find_pid_on_port(self.port) is not None:
            print(f"    ⚠ Port {self.port} still in use after "
                  f"{max_retries} kill attempts", flush=True)
            return False
        return True  # allow OS to release the port

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self, wait_up_to=120):
        """Start llama-server as a subprocess. Returns ``True`` on success."""
        if self.server_proc:
            # Already have a running subprocess — check it's alive
            if self.server_proc.poll() is None:
                return True
            self.server_proc = None

        # Kill any stale process still holding the port
        if not self._free_port():
            print(f"    ⚠ Port {self.port} still occupied — attempting start "
                  f"anyway", flush=True)

        # Clear any lingering CUDA allocations so the server gets
        # maximum available VRAM (important when preceding pipeline
        # stages — WhisperX, Wav2Vec2 — have released their models).
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except ImportError:
            pass

        # Re-detect GPU layers now that VRAM should be clean
        if self.gpu_layers is None:
            self.gpu_layers = auto_gpu_layers(self.model)
        if self.gpu_layers > 0:
            print(f"    GPU offload: {self.gpu_layers} layers", flush=True)

        cmd = [
            self.server_path, "-m", self.model_path,
            "--port", str(self.port),
            "-t", str(self.num_threads),
            "-c", str(self.context_size),
            "--cont-batching",
            "-ngl", str(self.gpu_layers),
        ]
        self.server_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self._server_pid = self.server_proc.pid
        # Register atexit safety-net (uses saved PID, not self.server_proc)
        if not self._cleanup_registered:
            import atexit
            atexit.register(self._kill_by_pid)
            self._cleanup_registered = True

        for i in range(wait_up_to):
            try:
                r = requests.get(f"{self.server_url}/health", timeout=2,
                                 proxies=NO_PROXY)
                if r.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        return False

    def stop(self):
        """Terminate llama-server with multi-layered shutdown.

        Bug fix: ``_kill_by_pid`` was checking ``self.server_proc``
        (already cleared) so Layer 3 was silently skipped.  Now the PID
        is saved before clearing and used directly in the fallback.
        Verifies after each layer that the port is actually free.
        """
        proc, self.server_proc = self.server_proc, None
        if proc is None:
            return

        pid = proc.pid  # save before proc ref may become invalid

        # Layer 1: terminate (TerminateProcess on Windows)
        print(f"    Stopping llama-server (PID {pid})…", flush=True)
        try:
            proc.terminate()
            proc.wait(timeout=5)
            # Verify the process actually exited
            if proc.poll() is not None:
                self._server_pid = None
                self._free_port()
                return
        except Exception:
            pass

        # Layer 2: kill (also TerminateProcess, but more emphatic)
        try:
            proc.kill()
            proc.wait(timeout=3)
            if proc.poll() is not None:
                self._server_pid = None
                self._free_port()
                return
        except Exception:
            pass

        # Layer 3: taskkill /F /PID on the saved PID
        # (self.server_proc is None at this point, so _kill_by_pid
        #  would return without doing anything — hence the direct call.)
        print(f"    Force-killing llama-server PID {pid}…", flush=True)
        self._kill_pid(pid, log_prefix="    ")
        self._server_pid = None
        time.sleep(0.5)
        self._free_port()

        # Layer 4: also try /T (process tree) if port is still occupied
        if self._find_pid_on_port(self.port) is not None:
            oops_pid = self._find_pid_on_port(self.port)
            print(f"    Port {self.port} still in use by PID {oops_pid}, "
                  f"killing tree…", flush=True)
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(oops_pid)],
                    capture_output=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                time.sleep(1)
            except Exception:
                pass

    def _kill_by_pid(self):
        """Kill llama-server by PID via ``taskkill /F /PID`` (Windows fallback).

        Uses ``_server_pid`` (saved at startup) rather than ``server_proc``
        so this works even after :meth:`stop` has cleared the process reference.
        """
        pid = self._server_pid
        if pid is None:
            return
        print(f"    [atexit] Cleaning up llama-server PID {pid}…", flush=True)
        self._kill_pid(pid, log_prefix="    [atexit] ")
        self.server_proc = None

    # ── LLM call ───────────────────────────────────────────────────────────

    def chat(self, system, user, max_tokens=None, temperature=0, cache_prompt=True):
        """Send prompt via ``/completion`` with the model-specific chat template.

        For the local backend *max_tokens* maps to ``n_predict``.
        """
        # Suppress thinking/reasoning for models that default to it
        if is_reasoning_model(self.model):
            system += (
                " IMPORTANT: Do NOT use any thinking or reasoning process. "
                "Do NOT output <think> tags or any meta-commentary. "
                "Just output the answer directly without any preamble."
            )
        prompt = _fmt_prompt(system, user, model=self.model)
        if max_tokens is None:
            n_pred = max(64, len(user) // 4 + 30)
        else:
            n_pred = max_tokens

        body = {
            "prompt": prompt,
            "n_predict": n_pred,
            "temperature": temperature,
            "cache_prompt": cache_prompt,
        }

        for attempt in range(2):
            try:
                r = requests.post(
                    f"{self.server_url}/completion",
                    json=body,
                    timeout=120,
                    proxies=NO_PROXY,
                )
                if r.status_code == 200:
                    r.encoding = "utf-8"
                    return r.json().get("content", "").strip()
                if r.status_code == 500 and attempt == 0:
                    # PEG-parser error — may still be salvageable on retry
                    continue
                break
            except Exception:
                break
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  OpenAI-compatible backend
# ══════════════════════════════════════════════════════════════════════════════


class LLMCallOpenAI(LLMCall):
    """OpenAI-compatible API backend (DeepSeek, OpenAI, Qwen, Gemini, Ollama …)."""

    def __init__(self, backend, model, api_key, base_url=""):
        self.backend = backend
        self.model = model
        cfg = BACKEND_DEFAULTS[backend]
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.api_key = api_key

    def chat(self, system, user, max_tokens=None, temperature=0, cache_prompt=True):
        """Send prompt via ``/chat/completions``.

        For API backends *cache_prompt* is ignored (not supported).
        """
        if max_tokens is None:
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
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # DeepSeek defaults to thinking mode which eats the token budget;
        # explicitly disable it when using DeepSeek.
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
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Anthropic backend
# ══════════════════════════════════════════════════════════════════════════════


class LLMCallAnthropic(LLMCall):
    """Anthropic Claude API backend."""

    def __init__(self, model, api_key, base_url=""):
        self.model = model
        cfg = BACKEND_DEFAULTS["anthropic"]
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.api_key = api_key

    def chat(self, system, user, max_tokens=None, temperature=0, cache_prompt=True):
        """Send prompt via the Anthropic Messages API (``/v1/messages``).

        *cache_prompt* is ignored (not supported by this backend).
        """
        if max_tokens is None:
            max_tokens = max(200, int(len(user) * 1.5))

        try:
            r = requests.post(
                f"{self.base_url}/v1/messages",
                json={
                    "model": self.model,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    "temperature": temperature,
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
                return ""
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Factory
# ══════════════════════════════════════════════════════════════════════════════


def create_llm_call(backend, model=None, api_key="", base_url="", gpu_layers=None):
    """Return an :class:`LLMCall` instance for the given *backend*.

    Args:
        backend: ``"local"``, ``"deepseek"``, ``"openai"``, ``"qwen"``,
                 ``"gemini"``, or ``"anthropic"``.
        model: model name (``None`` → per-backend default).
        api_key: API key for online backends.
        base_url: override the default base URL.
        gpu_layers: GPU offload layers (local backend only).

    Returns:
        :class:`LLMCallLocal`, :class:`LLMCallOpenAI`, or :class:`LLMCallAnthropic`.
    """
    if backend == "local":
        m = model or "phi4"
        return LLMCallLocal(model=m, gpu_layers=gpu_layers)

    if backend not in BACKEND_DEFAULTS:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            f"Available: local, {', '.join(BACKEND_DEFAULTS)}"
        )

    cfg = BACKEND_DEFAULTS[backend]
    m = model or cfg["default_model"]

    if backend == "anthropic":
        if not api_key:
            print("  [WARN] anthropic_api_key is not set")
        return LLMCallAnthropic(model=m, api_key=api_key, base_url=base_url)

    # Detect DeepSeek's Anthropic-compatible endpoint
    if backend == "deepseek":
        base = base_url or "https://api.deepseek.com"
        if "anthropic" in base.lower():
            return LLMCallAnthropic(model=m, api_key=api_key, base_url=base)

    # All other backends use the OpenAI-compatible format
    return LLMCallOpenAI(backend=backend, model=m, api_key=api_key, base_url=base_url)
