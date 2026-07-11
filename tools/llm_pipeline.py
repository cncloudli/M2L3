"""
LLM-powered punctuation restoration and sentence segmentation.

Uses a GGUF model via llama-server (GPU-accelerated) for subtitle segmentation.
Supports Phi-4 via the ``model`` parameter.

Architecture:
  1. Preserve native WhisperX punctuation (periods, commas, apostrophes)
  2. Split word list at native .?! into "groups" (natural sentence units)
  3. Consecutive overlength groups are merged into "blocks"
  4. Each block is sent to the LLM with surrounding context (read-only)
  5. Character-level diff between input and output to find ADDED .?!
  6. Map new break positions back to original word indices
  7. Split word list at native + new break points → segments
  8. Comma-based force-split of still-overlength segments (clause commas only)
"""

import os, re, subprocess, time, json, difflib, atexit
import sys
from pathlib import Path
import requests

from tools.non_split_bigrams import would_break_phrase

# ── PyInstaller-aware paths ──────────────────────────────────────────────────
# ``_APP_DIR`` points to where user data (models/, input/, output/) lives.
# ``_BUNDLE_DIR`` points to where bundled internal files (tools/llama/) live.
# In dev mode both are the project root; in packaged mode they diverge.
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
    _BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _APP_DIR = _BASE_DIR
    _BUNDLE_DIR = _BASE_DIR

LLAMA_SERVER = os.path.join(_BUNDLE_DIR, "tools", "llama", "llama-server.exe")
MODEL_PATH = os.path.join(_APP_DIR, "models", "phi-4-Q4_K_M.gguf")
NO_PROXY = {"http": None, "https": None}


def clean_llm_output(text):
    """Strip trailing meta-commentary (Note:, Explanation:, etc.) from LLM output."""
    for marker in (r"\bNote\s*:", r"\bExplanation\s*:", r"\bExample\s*:", r"\bDisclaimer\s*:"):
        m = re.split(marker, text, flags=re.IGNORECASE)
        text = m[0].strip()
    text = re.sub(r"\n\[.*?\].*", "", text).strip()
    return text


"""
Comma-based force-split for the LLM pipeline — replaces the old
_llm_force_split / _best_split gap-based approach.  
Only splits at commas where the following word is a clause-starter 
(pronoun, conjunction, WH-word, etc.) — NOT at list commas ("a, b and c").

Uses right-to-left single-split recursion: find the rightmost
qualifying comma, split there, and recurse on both sub-segments.
This guarantees no fragment is shorter than min_words.
"""

"""
so-intensifier detection — Words that, when they follow "so", 
indicate "so" is an intensifier (adverb meaning "very/that much") 
rather than a conjunction ("therefore").

Conjunction "so" is always followed by a clause subject (pronoun, noun, or demonstrative).  
Intensifier "so" is followed by an adjective, adverb, or quantifier.  
The set below covers short common targets; 
longer forms (-ly adverbs, -ful/-less adjectives, etc.) are handled dynamically 
by _is_so_intensifier_target().
"""
_SO_INTENSIFIER_ADJS = frozenset({
    # Quantifiers — extremely common in tutorials
    'much', 'many', 'little', 'few',
    # Core short adjectives
    'good', 'bad', 'big', 'small', 'long', 'short', 'high', 'low',
    'fast', 'slow', 'hard', 'soft', 'loud', 'quiet',
    'bright', 'dark', 'wide', 'deep', 'hot', 'cold', 'warm', 'cool',
    'great', 'huge', 'tiny', 'vast', 'large',
    'easy', 'simple', 'clear', 'clean', 'smooth', 'rough',
    'early', 'late', 'soon', 'far', 'close', 'near', 'often',
    'right', 'wrong', 'true', 'sure', 'certain',
    'new', 'old', 'young', 'fresh',
    'strong', 'weak', 'nice', 'kind', 'weird', 'strange',
    'happy', 'sad', 'calm',
    # Superlatives
    'best', 'worst', 'most', 'least',
    'biggest', 'smallest', 'greatest', 'largest', 'highest',
    'fastest', 'slowest', 'loudest', 'quietest', 'easiest',
    # Comparatives
    'better', 'worse', 'greater', 'larger', 'smaller',
    'higher', 'lower', 'faster', 'slower', 'earlier', 'later',
    'more', 'less',
})


def _is_so_intensifier_target(w):
    """Check if word *w* (lowercase, cleaned) after 'so' suggests 'so'
    is an intensifier (adverb) rather than a conjunction.

    Returns True for adjectives, adverbs, and quantifiers that typically
    follow intensifier 'so' in spoken English (e.g. 'so good', 'so much', 'so quickly').
    Split before these would break the intensifier phrase.
    """
    if w in _SO_INTENSIFIER_ADJS:
        return True
    # -ly adverbs — very reliable indicator of intensifier so
    if w.endswith('ly') and len(w) > 4:
        return True
    # Common adjective suffixes (captures -ful, -less, -ive, -able, etc.)
    if re.search(r'(?:ful|less|ous|ive|able|ible|ic|ical|esque|like|some|ish)$', w):
        return True
    return False


# Words that can start a new clause after a comma.
_CLAUSE_STARTERS = frozenset({
    # Subject pronouns
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    # Demonstratives / existential
    'this', 'that', 'these', 'those', 'there',
    # Coordinating conjunctions (introducing independent clauses)
    'and', 'but', 'or', 'so', 'nor', 'yet', 'for',
    # Subordinating conjunctions
    'if', 'when', 'because', 'although', 'since',
    'unless', 'though', 'while', 'where', 'whereas',
    'as', 'once', 'until',
    # WH-words
    'which', 'who', 'whom', 'whose', 'what', 'how', 'why',
    # Conditional / indirect-question marker
    'whether',
})

"""
Words that start elaboration / comparison / qualification phrases.
Unlike _CLAUSE_STARTERS, these are not subjects or conjunctions 
— they head adverbial or prepositional phrases that add detail to the 
preceding clause (e.g. "...bend these, very similar to MSEGs").  
These words never appear as list items ("a, b and c"), so splitting at them is safe.
"""
_ELABORATION_STARTERS = frozenset({
    # Comparison / similarity
    'very',       # "...bend these, very similar to MSEGs"
    'similar',    # "...the MSEGs, similar to envelopes"
    'much',       # "...the sound, much like a pad"
    'more',       # "...the bass, more of a sub sound"
    # Specification / exemplification
    'especially', # "...the filters, especially the low pass"
    'particularly',
    'including',  # "...features, including X, Y and Z"
    'such',       # "...reverb, such as hall or plate"
    'like',       # "...noise, like white or pink"
    'notably',    # "...the filters, notably the low pass"
    'namely',     # "...two effects, namely compression and reverb"
    # Scope / domain qualification
    'mostly',
    'mainly',
    'primarily',
    'largely',
    'typically',  # "...the frequency, typically around 20 Hz"
    'generally',  # "...the process, generally speaking"
    # Conditional / dependency
    'depending',  # "...noise, depending on what you want"
    'based',      # "...the routing, based on your settings"
    'compared',   # "...linear, compared to non-linear"
    'according',  # "...the manual, according to the specs"
    'excluding',  # "...the reverb, excluding the dry signal"
    'except',     # "...all settings, except for the filter"
    'related',    # "...the noise, related to the filter"
    # Modifying adverbs
    'essentially',
    'specifically',
})


# Words that, when they appear immediately after a comma, unambiguously
# start a new independent clause.  Lists never have items beginning with
# subject pronouns or demonstratives, so their presence means the comma
# is a clause boundary, not a list separator.
_CLAUSE_SUBJECTS = frozenset({
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'this', 'that', 'these', 'those', 'there',
})


def _is_list_comma(right_words):
    """Check if a comma separates list items via and/or in the right half.

    Scans the right half for the first ``and`` / ``or``.
    If it looks like a list connector (no clause signal nearby), 
    returns True → this is a list comma → don't split there.

    **Early-exit guard**: if the very first word after the comma is a subject pronoun/demonstrative (``_CLAUSE_SUBJECTS``)
    — e.g. *"hours,**we're** going to..."* — the comma is a clause boundary, not a list comma.
    Without this guard, ``_is_list_comma`` could scan *past* the clause starter and find a distant ``and/or`` far to the right, 
    incorrectly flagging a clause comma as a list comma.

    Shared by Phase 6 (``_comma_split``) and Phase 10 (``_emergency_comma_split``).
    """
    if right_words:
        first = right_words[0]
        # Resolve contractions: "we're" → "we", "it's" → "it"
        root = first.split("'")[0] if "'" in first else first
        if root in _CLAUSE_SUBJECTS:
            return False

    for idx, w in enumerate(right_words):
        if w not in ('and', 'or'):
            continue

        w1 = right_words[idx + 1] if idx + 1 < len(right_words) else None
        if w1 is None:
            continue
        if w1 in _CLAUSE_STARTERS:
            return False
        if w1 == 'to':
            return False

        w2 = right_words[idx + 2] if idx + 2 < len(right_words) else None
        if w2 is not None:
            if w2 in _CLAUSE_STARTERS:
                return False
            if w2 == 'to':
                return False

        # First and/or in right half has no clause signal → list
        return True

    # No and/or found → not a list comma (safe to split)
    return False


def _comma_split(seg_words, min_words=4):
    """Split an overlength segment at clause-internal commas (recursive).

    For each comma in the segment:
      - Left side (not counting the comma word itself) must have ≥ min_words
      - Right side (not counting the comma word itself) must have ≥ min_words
      - First word of right side must be a CLAUSE_STARTER (pronoun, conjunction, WH-word, etc.) 
        or an ELABORATION_STARTER (adverb / comparison / qualification word) — NOT a noun, verb, or list item.
      - Contractions are resolved to their root (e.g. ``it'll`` → ``it``).

    Algorithm (right-to-left, single-split recursion):
      Scan commas from rightmost to leftmost.  
      Take the first (rightmost) comma that satisfies all guards and split at it — creating exactly TWO sub-segments.  
      Then recurse on each sub-segment.
      This guarantees no resulting fragment is shorter than min_words:
      the multi-comma intermediate-segment problem (two adjacent split points with only 1–2 words between them) 
      cannot arise because each recursion level creates exactly one split → no intermediate gap.

    Args:
        seg_words: list of word dicts with ``text``, ``start``, ``end``.
        min_words: minimum words on each side of split (default 4).

    Returns:
        List of word-sublists (same format as the old ``_llm_force_split``).
    """
    n = len(seg_words)

    # Right-to-left scan — try the rightmost qualifying comma first.
    for i in range(n - 1, -1, -1):
        w = seg_words[i]
        if not w["text"].rstrip().endswith(','):
            continue
        # i = index of word ending with comma; split after this word
        # Comma word is NOT counted on either side (global policy).
        left_cnt = i               # words before the comma word
        right_cnt = n - i - 1      # words after the comma word
        if left_cnt < min_words or right_cnt < min_words:
            continue

        first_right = seg_words[i + 1]["text"].strip().lower().rstrip(',.')
        # Resolve contractions: "it'll" → "it", "we're" → "we"
        root = first_right.split("'")[0] if "'" in first_right else first_right
        if root not in _CLAUSE_STARTERS and root not in _ELABORATION_STARTERS:
            continue

        # Even if the right-side starter seems safe, check whether this
        # comma is actually a list separator (e.g. ``fast, very fast``
        # where ``very`` would be flagged as an elaboration starter).
        # If the right half has a list-structure and/or, don't split here.
        right_words = []
        for j in range(i + 1, n):
            wt = seg_words[j]["text"].strip()
            right_words.append(wt.lower().rstrip('.,!?;\'"'))
            # Truncate at the next comma — a list connector (and/or)
            # should appear before the next comma in the right half.
            # Scanning past the next comma risks attributing a distant
            # and/or (that belongs to a different comma) to this one.
            if wt.endswith(','):
                break
        is_list = _is_list_comma(right_words)
        print(f"    Phase 6 comma idx={i} word=\"{w['text']}\" "
              f"right_words={right_words} is_list={is_list}", flush=True)
        if is_list:
            continue

        # Found the rightmost qualifying comma — split ONE comma per call,
        # then recurse on each side.  The min_words guard above guarantees
        # both sub-segments are long enough.
        split_idx = i + 1  # exclusive-end index (comma word stays with left)
        left_part = seg_words[:split_idx]
        right_part = seg_words[split_idx:]

        result = []
        result.extend(_comma_split(left_part, min_words))
        result.extend(_comma_split(right_part, min_words))
        return result

    # No qualifying comma found
    return [seg_words]




"""
Conjunction-based force-split for LLM pipeline (Phase 7) 
— Phase 7 targets segments that survived Phase 6 (no splittable comma) but are still overlength.
It splits at coordinating conjunctions (and, but, or, so) that introduce new independent clauses rather than listing items.

Tier 1 (rules, no LLM required):
  - but / so / or → unconditionally splittable (with min_words guard)
  - and + CLAUSE_STARTER (pronoun, WH-word, etc.) → splittable

Tier 2 (LLM-assisted):
  - and + non-CLAUSE_STARTER → ask LLM to classify

Uses the same right-to-left single-split recursion as _comma_split.
"""

# Words treated as clause-level coordinating conjunctions.
_CONJUNCTIONS = frozenset({'and', 'but', 'or', 'so'})


def _find_ambiguous_conjunctions(seg_words, min_words=4):
    """Find conjunction positions (so/or/and) NOT covered by the rule layer.

    Returns word indices where the conjunction is followed by a non-CLAUSE_STARTER,
    meaning the rule layer cannot decide and LLM confirmation is needed.

    Conjunctions followed by CLAUSE_STARTER (e.g. ``so I``, ``or we``, ``and you``) are handled by Tier 1 rules.  
    Intensifier ``so`` (so + adj/adv) is also excluded — it is never a valid split point.
    """
    n = len(seg_words)
    positions = []
    for i, w in enumerate(seg_words):
        word = w["text"].rstrip(',.').lower()
        if word not in ('and', 'so', 'or'):
            continue
        left_count = i
        right_count = n - i
        if left_count < min_words or right_count < min_words:
            continue
        if i + 1 < n:
            first_right = seg_words[i + 1]["text"].strip().lower().rstrip(',.')
            root = first_right.split("'")[0] if "'" in first_right else first_right
            # CLAUSE_STARTER after conjunction → Tier 1 rule handles it
            if root in _CLAUSE_STARTERS:
                continue
            # Intensifier so + adj/adv → never ambiguous, never a split point
            if word == 'so' and _is_so_intensifier_target(root):
                continue
        positions.append(i)
    return positions


def _conjunction_split(seg_words, min_words=4, llm_confirmed_ids=None,
                       list_conjunctions=None):
    """Split an overlength segment at clause-level conjunctions (recursive).

    Right-to-left scan.  For each conjunction (and/but/or/so) that passes the
    min_words guard:
      - but → always split (these are almost always clause-level).
      - so + CLAUSE_STARTER → split (conjunction, rule-based).
      - so + adj/adv (intensifier) → never split.
      - so + other → split only if LLM-confirmed (Tier 2).
      - or + CLAUSE_STARTER → split (rule-based).
      - or + other → split only if LLM-confirmed (Tier 2).
      - and + CLAUSE_STARTER → split (rule-based).
      - and + other → split only if LLM-confirmed (Tier 2).

    Split happens BEFORE the conjunction → the conjunction attaches to the right sub-segment, 
    which reads more naturally in subtitles.

    Args:
        seg_words: list of word dicts.
        min_words: minimum words on each side (default 4).
        llm_confirmed_ids: set of ``id(word)`` for "and" positions that the
                           LLM confirmed as clause-level.

    Returns:
        List of word-sublists (same format as ``_comma_split``).
    """
    n = len(seg_words)

    # Right-to-left scan — try the rightmost qualifying conjunction first.
    for i in range(n - 1, -1, -1):
        word_text = seg_words[i]["text"].rstrip(',.').lower()
        if word_text not in _CONJUNCTIONS:
            continue

        left_count = i
        right_count = n - i
        if left_count < min_words or right_count < min_words:
            continue

        can_split = False

        if word_text == 'but':
            # Tier 1 — always splittable (but is almost always clause-level)
            can_split = True
        elif word_text == 'so':
            first_right = seg_words[i + 1]["text"].strip().lower().rstrip(',.')
            root = first_right.split("'")[0] if "'" in first_right else first_right
            if root in _CLAUSE_STARTERS:
                # Tier 1 — subject follows → conjunction so
                can_split = True
            elif _is_so_intensifier_target(root):
                # Intensifier so (adverb: "very / that much") — never split.
                # E.g. "so good", "so much", "so quickly".
                can_split = False
            elif llm_confirmed_ids and id(seg_words[i]) in llm_confirmed_ids:
                # Tier 2 — LLM-confirmed
                can_split = True
        elif word_text == 'or':
            first_right = seg_words[i + 1]["text"].strip().lower().rstrip(',.')
            root = first_right.split("'")[0] if "'" in first_right else first_right
            if root in _CLAUSE_STARTERS:
                # Tier 1 — subject follows → or opening a clause
                can_split = True
            elif llm_confirmed_ids and id(seg_words[i]) in llm_confirmed_ids:
                # Tier 2 — LLM-confirmed
                can_split = True
        elif word_text == 'and':
            first_right = seg_words[i + 1]["text"].strip().lower().rstrip(',.')
            root = first_right.split("'")[0] if "'" in first_right else first_right
            if root in _CLAUSE_STARTERS:
                # Tier 1 — rule-based
                can_split = True
            elif llm_confirmed_ids and id(seg_words[i]) in llm_confirmed_ids:
                # Tier 2 — LLM-confirmed
                can_split = True

        if can_split:
            # Reject split at protected bigrams (Phase 7 must not break
            # fixed expressions like "just so", "such as", etc.)
            if would_break_phrase(seg_words, i):
                continue
            """
            List-enumeration guard: reject and/or splits when the right half 
            looks like a list continuation (no clause-starting word).
            Prevents breaking "parameters, buttons, and knobs" at "and".

            Uses PRE-SCANNED list_conjunctions (set of id() values) rather than 
            calling _is_list_comma() at runtime, 
            because the recursive right-to-left split truncates right_words
            and the second and/or that signals a list may already be in a different sub-segment.
            """
            if word_text in ('and', 'or'):
                if list_conjunctions and id(seg_words[i]) in list_conjunctions:
                    print(f"      Phase 7 BLOCK list comma (pre-scanned): "
                          f"i={i} word=\"{word_text}\"", flush=True)
                    continue
            # Split BEFORE the conjunction (conjunction goes to right segment)
            left_part = seg_words[:i]
            right_part = seg_words[i:]

            print(f"      Phase 7 SPLIT: i={i} word=\"{word_text}\" "
                  f"can_split={can_split}", flush=True)

            result = []
            result.extend(_conjunction_split(left_part, min_words, llm_confirmed_ids,
                                               list_conjunctions))
            result.extend(_conjunction_split(right_part, min_words, llm_confirmed_ids,
                                               list_conjunctions))
            return result

    return [seg_words]


# ── Model registry ────────────────────────────────────────────────────────────
# Map model names to their GGUF file in models/ directory.
MODEL_REGISTRY = {
    "phi4":      "phi-4-Q4_K_M.gguf",
}

# Known layer counts for auto GPU offload calculation.
_MODEL_LAYERS = {
    "phi4":      40,   # Phi-4 14.7B
}


class LLMPipeline:
    """Punctuation restoration pipeline via llama-server.

    Supports the Phi-4 model via the ``model`` parameter.
    """

    """
    Fragile trailing words — Words that, when they appear at the end of a segment, indicate a BAD split point.  
    These cover articles, possessives, auxiliaries, prepositions, subordinating/coordinating conjunctions, 
    bare adverbs, quantifiers, and other function words that grammatically depend on what follows.
    """
    FRAGILE_TRAILING = frozenset({
        # Articles & determiners
        'the', 'a', 'an',
        # Demonstratives
        'this', 'that', 'these', 'those', 'same',
        # Possessives
        'my', 'your', 'his', 'her', 'its', 'our', 'their',
        # Modals
        'will', 'would', 'can', 'could', 'shall', 'should',
        'may', 'might', 'must',
        # To-be verbs
        'is', 'am', 'are', 'was', 'were', 'been', 'being',
        # Perfect/have auxiliaries
        'have', 'has', 'had',
        # Do-support
        'do', 'does', 'did',
        # Prepositions
        'in', 'on', 'at', 'for', 'with', 'by', 'from', 'of', 'about',
        'to', 'into', 'through', 'during', 'without', 'between',
        'under', 'over', 'before', 'after', 'above', 'below',
        'upon', 'onto', 'against', 'among', 'beside',
        'beneath', 'underneath', 'around', 'behind', 'along',
        'across', 'toward', 'towards', 'throughout',
        'inside', 'outside', 'via', 'per', 'near',
        'despite', 'within', 'unlike', 'versus',
        # Subordinating conjunctions
        'if', 'when', 'while', 'because', 'although', 'since',
        'unless', 'though', 'whereas', 'how',
        'as', 'once', 'until',
        # Relative pronouns
        'which', 'who', 'whom', 'whose', 'where',
        # Coordinating conjunctions
        'and', 'but', 'or', 'so', 'nor', 'yet',
        # Bare adverbs (modify the next verb)
        'just', 'almost', 'nearly', 'really', 'barely', 'hardly',
        'merely', 'simply', 'quite', 'virtually', 'essentially',
        'practically', 'roughly', 'approximately',
        'already', 'also', 'still',
        # Fixed constructions
        'going',  # be going to — split at "going" breaks the future-construction
        # Negatives / qualifiers
        'no', 'not', 'even', 'too', 'very', 'rather', 'only',
        # Quantifiers
        'some', 'any', 'many', 'much', 'more', 'most', 'few',
        'several', 'each', 'every', 'both', 'all',
        # Determiners / indefinite adjectives
        'such',
    })

    # Pre-compiled regex: does text end with a FRAGILE_TRAILING word?
    _FRAGILE_RE = re.compile(
        r'\b(' + '|'.join(
            re.escape(w)
            for w in sorted(FRAGILE_TRAILING, key=len, reverse=True)
        ) + r')\s*$',
        re.I,
    )

    def __init__(self, model="phi4", model_path=None, server_path=None,
                 port=8088, num_threads=12, context_size=4096, gpu_layers=None):
        self.model = model
        if model_path is not None:
            self.model_path = model_path
        elif model in MODEL_REGISTRY:
            self.model_path = os.path.join(_APP_DIR, "models",
                                           MODEL_REGISTRY[model])
        else:
            self.model_path = MODEL_PATH  # fallback to module default
        self.server_path = server_path or LLAMA_SERVER
        self.port = port
        self.server_url = f"http://127.0.0.1:{port}"
        self.num_threads = num_threads
        self.context_size = context_size
        self.gpu_layers = gpu_layers  # None means auto-detect
        self.server_proc = None

    # ── Auto GPU layers ────────────────────────────────────────────────────

    def _auto_gpu_layers(self):
        """Automatically determine how many model layers to offload to GPU
        based on available VRAM and model file size.

        Uses a fixed 1.5 GB reservation for CUDA context + KV cache + safety buffer.  
        The remaining VRAM is divided by the estimated per-layer cost
        (model file size / total layers × 10% overhead).

        Returns:
            int: number of layers to offload (0 if CUDA is unavailable).
        """
        try:
            import torch
        except ImportError:
            return 0
        if not torch.cuda.is_available():
            return 0
        try:
            props = torch.cuda.get_device_properties(0)
            total_vram = props.total_memory / 1e9  # GB
            model_size = os.path.getsize(self.model_path) / 1e9  # GB
            n_layers = _MODEL_LAYERS.get(self.model, 0)
            if n_layers == 0:
                return 0
            # Reserve 1.5 GB for CUDA context + KV cache + safety buffer
            available = total_vram - 1.5
            if available <= 0:
                return 0
            per_layer = model_size / n_layers * 1.1  # 10 % overhead
            return max(0, min(n_layers, int(available / per_layer)))
        except Exception:
            return 0

    # ── Prompt format helpers ──────────────────────────────────────────────

    def _fmt_prompt(self, system, user):
        """Wrap system + user content in the Phi-4 chat template.

        Args:
            system: system message string.
            user: user message string.

        Returns:
            Formatted prompt ready for llama-server ``/completion``.
        """
        return (
            f"<|system|>\n{system}\n<|end|>\n"
            f"<|user|>\n{user}\n<|end|>\n"
            f"<|assistant|>\n"
        )

    # ── Server lifecycle ──────────────────────────────────────────────────

    def start(self, wait_up_to=120):
        """Start llama-server as a subprocess. Returns True on success."""
        if self.server_proc:
            return True  # already running

        # Auto-detect GPU layers if not explicitly set
        if self.gpu_layers is None:
            self.gpu_layers = self._auto_gpu_layers()
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
        print(f"  Starting {self.model.upper()} (llama-server)...", end=" ", flush=True)
        self.server_proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        # Register cleanup as atexit safety net (will run if process exits
        # without going through stop(), e.g. sys.exit() or unhandled crash).
        # Guard against double-registration via _cleanup_registered flag.
        if not getattr(self, '_cleanup_registered', False):
            atexit.register(self._kill_by_pid)
            self._cleanup_registered = True

        for i in range(wait_up_to):
            try:
                r = requests.get(f"{self.server_url}/health", timeout=2, proxies=NO_PROXY)
                if r.status_code == 200:
                    print(f"ok ({i+1}s)")
                    return True
            except requests.RequestException:
                pass
            time.sleep(1)
        print("FAIL")
        return False

    def _kill_by_pid(self):
        """Kill llama-server by PID using taskkill (Windows-native fallback).

        This is also registered as an atexit handler for cases 
        where the normal stop() path is skipped (e.g. sys.exit, unhandled crash).
        Guarded so it's safe to call multiple times.
        """
        proc = getattr(self, 'server_proc', None)
        if proc is None:
            return
        pid = proc.pid
        # taskkill /F is the most reliable Windows process kill
        try:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            print(f"  [OK] llama-server (PID {pid}) killed via taskkill", flush=True)
        except Exception as e:
            print(f"  [WARN] taskkill failed for PID {pid}: {e}", flush=True)
        self.server_proc = None

    def stop(self):
        """Terminate llama-server subprocess. Multi-layered approach:

        1. terminate() + wait (gentle, may not work on all Windows builds)
        2. kill() if terminate() times out
        3. taskkill /F /PID as final Windows-native fallback
        4. blanket taskkill /IM cleans up orphans from prior sessions
        """
        if self.server_proc:
            pid = self.server_proc.pid
            print(f"  Shutting down llama-server (PID {pid})...", end=" ", flush=True)

            # Layer 1: terminate()
            try:
                self.server_proc.terminate()
                self.server_proc.wait(timeout=5)
                print(f"ok (terminate)", flush=True)
                self.server_proc = None
                return
            except subprocess.TimeoutExpired:
                print(f"timeout ", end="", flush=True)
            except Exception as e:
                print(f"err({e}) ", end="", flush=True)

            # Layer 2: kill()
            try:
                self.server_proc.kill()
                self.server_proc.wait(timeout=3)
                print(f"ok (kill)", flush=True)
                self.server_proc = None
                return
            except Exception as e:
                print(f"kill-err({e}) ", end="", flush=True)

            # Layer 3: taskkill /F /PID
            print(f"fallback(taskkill)...", end=" ", flush=True)
            self._kill_by_pid()

        # Layer 4: blanket cleanup of any orphaned llama-server instances
        # from prior sessions (harmless if none are running).
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", "llama-server.exe"],
                capture_output=True, timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            if result.returncode == 0:
                stdout = result.stdout.decode('utf-8', errors='replace').strip()
                print(f"  [OK] Orphaned llama-server.exe: {stdout}", flush=True)
        except Exception:
            pass

    @property
    def is_running(self):
        if not self.server_proc:
            return False
        return self.server_proc.poll() is None

    # ── LLM request ──────────────────────────────────────────────────────

    def _llm(self, text, left_context=None, right_context=None, max_retries=2,
             system_override=None):
        """Send text to LLM for punctuation, with optional surrounding context.

        The system prompt tells the LLM to preserve existing punctuation 
        and only add missing commas/sentence-ending marks.  
        When *left_context* or *right_context* is provided, 
        it is embedded in the system prompt as read-only information 
        so the LLM can make better break-point decisions without modifying surrounding sentences.

        Pass *system_override* to replace the entire default system prompt
        (used by Phase 8 for targeted run-on sentence re-punctuation).

        Args:
            text: the main text to punctuate (within the writable block).
            left_context: preceding sentence (read-only, for context).
            right_context: following sentence (read-only, for context).
            max_retries: number of LLM retries on failure.
            system_override: if provided, use this as the full system prompt
                             instead of the default one.

        Returns:
            Punctuated text string, or None after exhausting retries.
        """
        if system_override:
            system = system_override
        else:
            system = (
                "Fix punctuation in the ASR transcript below. Keep existing "
                "punctuation that is correct, but ADD missing commas and "
                "sentence-ending punctuation (. ? !) wherever natural reading "
                "requires them — especially periods between complete thoughts.\n"
                "\n"
                "⚠ WARNING: When items are listed as \"X and Y and Z\", the "
                "'and's are list-internal connectors. Do NOT add periods at "
                "them. Only split at 'and'/'but'/'or' if both sides are clear "
                "independent clauses with their own subjects and verbs.\n"
                "\n"
                "⚠ CRITICAL: NEVER add a period after a bare adverb like "
                "'just', 'almost', 'nearly', 'really', 'barely', 'hardly', "
                "'merely', 'simply', 'quite', 'virtually', 'essentially', "
                "'practically', 'roughly', 'approximately'. These adverbs "
                "modify the next verb and the sentence is NOT complete. "
                "Example: \"It's less intense and it just\" → DO NOT add a "
                "period after \"just\" because it needs a verb.\n"
                "\n"
            )
            if left_context or right_context:
                system += (
                    "For context only — the following sentences surround the "
                    "text you need to punctuate. Use them to determine natural "
                    "break points, but do NOT modify them.\n"
                )
                if left_context:
                    system += f"PRECEDING (read-only context): \"{left_context}\"\n"
                if right_context:
                    system += f"FOLLOWING (read-only context): \"{right_context}\"\n"
                system += "\nNow punctuate the MAIN TEXT below.\n"

            system += (
                "Never change wording. Never add explanations or meta-commentary. "
                "Output ONLY the punctuated text."
            )

        prompt = self._fmt_prompt(system=system, user=text)
        n_pred = max(64, len(text) // 4 + 30)
        for attempt in range(max_retries + 1):
            try:
                r = requests.post(
                    f"{self.server_url}/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": n_pred,
                        "temperature": 0,
                        "cache_prompt": True,
                    },
                    timeout=120,
                    proxies=NO_PROXY,
                )
                if r.status_code == 200:
                    raw = r.json().get("content", "").strip()
                    return clean_llm_output(raw)
                else:
                    print(f"    HTTP {r.status_code}", end=" ", flush=True)
            except Exception as e:
                print(f"    [{attempt+1}/{max_retries+1}] {e}", end=" ", flush=True)
            if attempt < max_retries:
                time.sleep(5)
        return None

    # ── Phase 7: 'and' classifier (LLM-assisted) ──────────────────────────

    def _classify_conjunctions(self, text, con_positions, seg_words):
        """Ask the LLM which conjunction positions connect two complete clauses.

        Works for ``and``, ``so``, and ``or``.  
        Each position is shown with the actual word 
        so the LLM can decide based on context 
        whether it introduces a new independent clause (YES) or connects list items / modifies an adverb (NO).

        Uses a simple yes/no prompt per batch of positions within one text.
        At temperature=0 the LLM should output a clean comma-separated list of YES/NO tokens.

        Args:
            text: the full text of the segment (string).
            con_positions: list of 1-based word indices to classify.
            seg_words: the segment's word-dict list (from which the actual
                       word text is extracted).

        Returns:
            list of bool in the same order as *con_positions*, or None on
            failure.
        """
        if not con_positions:
            return None

        items = "\n".join(
            f"  '{seg_words[p - 1]['text'].rstrip(',.').lower()}' at word position {p}"
            for p in con_positions
        )
        system = (
            "Is each word below connecting two complete clauses (each with "
            "its own subject and verb)? Answer ONLY with a comma-separated "
            'list of YES or NO, e.g. "YES, NO, YES".'
        )
        user = f"Text: \"{text}\"\n{items}"
        prompt = self._fmt_prompt(system=system, user=user)
        n_pred = max(8, len(con_positions) * 6)

        for attempt in range(2):
            try:
                r = requests.post(
                    f"{self.server_url}/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": n_pred,
                        "temperature": 0,
                        "cache_prompt": True,
                    },
                    timeout=30,
                    proxies=NO_PROXY,
                )
                if r.status_code == 200:
                    raw = r.json().get("content", "").strip()
                    tokens = [t.strip().upper().rstrip(".,!?")
                              for t in raw.replace(",", " ").split()]
                    results = [t == "YES" for t in tokens if t in ("YES", "NO")]
                    if len(results) == len(con_positions):
                        return results
                    # Fallback: line-by-line scan
                    results = []
                    for line in raw.replace(",", "\n").split("\n"):
                        uline = line.strip().upper()
                        if uline == "YES":
                            results.append(True)
                        elif uline == "NO":
                            results.append(False)
                    if len(results) == len(con_positions):
                        return results
                else:
                    print(f"    [Phase 7] LLM HTTP {r.status_code}", flush=True)
            except Exception as e:
                print(f"    [Phase 7] LLM error: {e}", flush=True)
            if attempt == 0:
                time.sleep(2)
        return None

    def _classify_conj_merge(self, items):
        """Classify whether conjunction-led segments are continuations.

        For each (prev_text, curr_text) pair, 
        determine whether the conjunction-led current segment 
        is a CONTINUATION of the previous sentence (should merge backward) or a NEW_SENTENCE (should not).

        Args:
            items: list of (prev_text, curr_text) tuples.

        Returns:
            list of bool in the same order as *items*:
              True  → CONTINUATION (merge backward recommended)
              False → NEW_SENTENCE (keep separate)
            Returns None on LLM failure.
        """
        if not items:
            return []

        item_lines = [
            f"  Item {i+1}:\n    Previous: \"{p}\"\n    Current: \"{c}\""
            for i, (p, c) in enumerate(items)
        ]
        user_text = "\n".join(item_lines)

        system = (
            "For each conjunction-led segment below, determine whether it "
            "is a CONTINUATION of the previous sentence or a NEW_SENTENCE.\n"
            "\n"
            "A CONTINUATION means the conjunction-led segment is "
            "grammatically and semantically a direct continuation of the "
            "previous segment. The conjunction (and/but/so/or) connects "
            "to the previous thought as a list item or elaboration.\n"
            "\n"
            "A NEW_SENTENCE means the conjunction starts a new independent "
            "thought that could stand alone as a sentence, even though it "
            "begins with a conjunction for rhetorical flow.\n"
            "\n"
            "Answer ONLY with a comma-separated list of CONTINUATION or "
            "NEW_SENTENCE, e.g. \"CONTINUATION, NEW_SENTENCE, CONTINUATION\"."
        )
        prompt = self._fmt_prompt(system=system, user=user_text)
        n_pred = max(8, len(items) * 14)

        for attempt in range(2):
            try:
                r = requests.post(
                    f"{self.server_url}/completion",
                    json={
                        "prompt": prompt,
                        "n_predict": n_pred,
                        "temperature": 0,
                        "cache_prompt": True,
                    },
                    timeout=60,
                    proxies=NO_PROXY,
                )
                if r.status_code == 200:
                    raw = r.json().get("content", "").strip()
                    # Split on comma or newline
                    tokens = [
                        t.strip().upper().rstrip(".,!?")
                        for t in raw.replace(",", " ").split()
                    ]
                    results = [
                        t == "CONTINUATION"
                        for t in tokens
                        if t in ("CONTINUATION", "NEW_SENTENCE")
                    ]
                    if len(results) == len(items):
                        return results
                    # Fallback: line-by-line scan
                    results = []
                    for line in raw.replace(",", "\n").split("\n"):
                        uline = line.strip().upper()
                        if uline == "CONTINUATION":
                            results.append(True)
                        elif uline == "NEW_SENTENCE":
                            results.append(False)
                    if len(results) == len(items):
                        return results
                else:
                    print(f"    [Phase 9 merge] LLM HTTP {r.status_code}",
                          flush=True)
            except Exception as e:
                print(f"    [Phase 9 merge] LLM error: {e}", flush=True)
            if attempt == 0:
                time.sleep(2)
        return None

    # ── Group-by-punctuation ───────────────────────────────────────────

    @staticmethod
    def _build_groups(words):
        """Split word indices at native .?! boundaries.

        Each group ends with (inclusive) a word whose stripped text
        carries native sentence-ending punctuation (. ? !).
        Groups with no .?! at all are kept as one unit.
        """
        groups = []
        start = 0
        for i, w in enumerate(words):
            if w["text"].rstrip().endswith(('.', '?', '!')):
                groups.append((start, i + 1))
                start = i + 1
        if start < len(words):
            groups.append((start, len(words)))
        return groups

    # ── Context resolution for long-group blocks ───────────────────────

    @staticmethod
    def _find_context(block_groups, sorted_groups, long_group_set):
        """Find nearest short groups before/after a block of long groups.

        Given a block (list of consecutive long-group ranges), 
        locate the nearest short group before (left context) and after (right context).
        These serve as read-only context when sending the block to the LLM.

        Args:
            block_groups: list of (start, end) for consecutive long groups.
            sorted_groups: sorted list of ALL (start, end) group ranges.
            long_group_set: set of (start, end) for overlength groups.

        Returns:
            (left_start, left_end) or None, (right_start, right_end) or None.
        """
        block_start = block_groups[0][0]
        block_end = block_groups[-1][1]

        left_ctx = None
        right_ctx = None

        # Iterate forward: each short group before block_start overwrites
        # the previous one, so we end up with the closest left context.
        for gs, ge in sorted_groups:
            if ge <= block_start and (gs, ge) not in long_group_set:
                left_ctx = (gs, ge)
            if gs >= block_end and (gs, ge) not in long_group_set and right_ctx is None:
                right_ctx = (gs, ge)

        return left_ctx, right_ctx

    # ── Character-diff based break detection ──────────────────────────

    @staticmethod
    def _build_char_to_word(raw_text):
        """Map character positions in *raw_text* to word indices."""
        mapping = {}
        word_idx = 0
        for i, ch in enumerate(raw_text):
            if ch == ' ':
                word_idx += 1
            else:
                mapping[i] = word_idx
        return mapping

    @staticmethod
    def _find_new_breaks(raw_text, punct_text, chunk_start, chunk_words):
        """Find word indices where the LLM added .?! not in native text.

        1. Find ALL .?! positions in both input and output.
        2. Use character-level SequenceMatcher to determine which output
           .?! positions map back to an input .?! (native) and which
           don't (new).
        3. Map each new .?! output-position → preceding-character's input
           position → word index.
        4. Return sorted list of exclusive-end word indices.
        """
        native_positions = {m.start() for m in re.finditer(r'[.?!]', raw_text)}
        output_positions = {m.start() for m in re.finditer(r'[.?!]', punct_text)}

        if not output_positions:
            return []

        # Map output → input char positions
        sm = difflib.SequenceMatcher(None, raw_text, punct_text)
        out_to_in = {}
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if tag == 'equal':
                for d in range(b2 - b1):
                    out_to_in[b1 + d] = a1 + d

        # Identify NEW .?! (not matching a native .?!)
        new_punct = set()
        for b_pos in output_positions:
            if b_pos in out_to_in:
                a_pos = out_to_in[b_pos]
                if a_pos < len(raw_text) and raw_text[a_pos] in '.?!':
                    continue  # native
            new_punct.add(b_pos)

        if not new_punct:
            return []

        char_to_word = LLMPipeline._build_char_to_word(raw_text)

        new_breaks = set()
        for b_pos in new_punct:
            # Walk back from punctuation to find a mapped character
            preceding = b_pos - 1
            while preceding >= 0 and preceding not in out_to_in:
                preceding -= 1
            if preceding < 0:
                continue

            in_pos = out_to_in[preceding]
            word_idx = char_to_word.get(in_pos)
            if word_idx is not None and word_idx < len(chunk_words):
                new_breaks.add(chunk_start + word_idx + 1)

        return sorted(new_breaks)

    @staticmethod
    def _find_new_commas(raw_text, punct_text, chunk_start, chunk_words):
        """Find word indices where the LLM added commas not in native text.

        Returns a set of global word indices that should get a comma appended.
        """
        native_positions = {m.start() for m in re.finditer(r',', raw_text)}
        output_positions = {m.start() for m in re.finditer(r',', punct_text)}

        if not output_positions:
            return set()

        # Map output → input char positions
        sm = difflib.SequenceMatcher(None, raw_text, punct_text)
        out_to_in = {}
        for tag, a1, a2, b1, b2 in sm.get_opcodes():
            if tag == 'equal':
                for d in range(b2 - b1):
                    out_to_in[b1 + d] = a1 + d

        # Identify NEW commas (not matching a native comma)
        new_commas = set()
        for b_pos in output_positions:
            if b_pos in out_to_in:
                a_pos = out_to_in[b_pos]
                if a_pos < len(raw_text) and raw_text[a_pos] == ',':
                    continue  # native
            new_commas.add(b_pos)

        if not new_commas:
            return set()

        char_to_word = LLMPipeline._build_char_to_word(raw_text)

        comma_word_indices = set()
        for b_pos in new_commas:
            preceding = b_pos - 1
            while preceding >= 0 and preceding not in out_to_in:
                preceding -= 1
            if preceding < 0:
                continue
            in_pos = out_to_in[preceding]
            word_idx = char_to_word.get(in_pos)
            if word_idx is not None and word_idx < len(chunk_words):
                comma_word_indices.add(chunk_start + word_idx)

        return comma_word_indices

    # ── Main entry point ────────────────────────────────────────────────

    def segment(self, words, max_chars=120, max_dur=9.0, max_words=30,
                min_words=4, min_dur=1.5):
        """Segment words into sentences via LLM punctuation filling.

        Architecture:
          1. Split at native .?! → groups (natural sentence units)
          2. Mark groups that exceed size limits (long groups)
          3. Merge consecutive long groups into "blocks"
          4. For each block, find read-only context (nearest short groups)
          5. Send block text to LLM with context in the system prompt
          6. Diff input vs output to find NEW .?! positions
          7. Programmatic FRAGILE_RE filter on new breaks
          8. Build segments from native + new breaks
          9. Gentle force-split of still-overlength segments

        Context groups are READ-ONLY — the LLM only adds punctuation within the writable block.  
        This prevents rules from leaking across independently-valid segment boundaries.

        Note: Word timestamp fixing is done upstream in main.py before this function is called.
        """
        # Lazy-start llama-server if not already running (started after
        # WhisperX transcription + Wav2Vec2 alignment to avoid VRAM contention).
        if not self.server_proc:
            if not self.start():
                print("[ERR] LLM server failed to start — native breaks only",
                      flush=True)
                # Fall through to native-breaks-only path below
        if not words:
            return []

        print(f"  {self.model.upper()} punctuation fill: {len(words)} words", flush=True)

        # ── Phase 1: Native groups at .?! ──────────────────────────────────
        groups = self._build_groups(words)
        print(f"    {len(groups)} native groups", flush=True)

        # ── Phase 2: Identify long groups ──────────────────────────────────
        long_group_set = set()
        for gs, ge in groups:
            span = words[gs:ge]
            dur = span[-1]['end'] - span[0]['start'] if span else 0
            if (ge - gs > max_words
                    or sum(len(words[j]["text"]) + 1 for j in range(gs, ge)) > max_chars
                    or dur > max_dur):
                long_group_set.add((gs, ge))

        n_long = len(long_group_set)
        n_short = len(groups) - n_long
        print(f"    {n_short} short, {n_long} long groups", flush=True)

        if not long_group_set:
            # Fast path: all groups short → native breaks suffice
            sorted_breaks = sorted(set(
                i + 1 for i, w in enumerate(words)
                if w["text"].rstrip().endswith(('.', '?', '!')))
                + [len(words)]
            )
            result = []
            prev = 0
            for b in sorted_breaks:
                if b <= prev:
                    continue
                seg_words = words[prev:b]
                text = " ".join(w["text"].strip() for w in seg_words).strip()
                if text:
                    result.append({"text": text, "start": seg_words[0]["start"],
                                   "end": seg_words[-1]["end"]})
                prev = b
            return result

        # ── Phase 3: Build blocks of consecutive long groups ──────────────
        sorted_groups = sorted(groups, key=lambda x: x[0])
        blocks = []
        i = 0
        while i < len(sorted_groups):
            gs, ge = sorted_groups[i]
            if (gs, ge) in long_group_set:
                block = [(gs, ge)]
                i += 1
                while i < len(sorted_groups):
                    gs2, ge2 = sorted_groups[i]
                    if (gs2, ge2) not in long_group_set:
                        break
                    block.append((gs2, ge2))
                    i += 1
                blocks.append(block)
            else:
                i += 1

        # Collect ALL native .?! positions as baseline breaks
        all_breaks = set()
        for iw, w in enumerate(words):
            if w["text"].rstrip().endswith(('.', '?', '!')):
                all_breaks.add(iw + 1)

        # ── Phase 4: Process each block through LLM with context ──────────
        for bi, block_groups in enumerate(blocks):
            block_start = block_groups[0][0]
            block_end = block_groups[-1][1]
            block_words = words[block_start:block_end]
            raw_text = " ".join(w["text"].strip() for w in block_words)

            # Find read-only context (nearest short groups)
            left_ctx, right_ctx = self._find_context(
                block_groups, sorted_groups, long_group_set)

            left_text = None
            if left_ctx:
                left_text = " ".join(
                    words[j]["text"].strip() for j in range(left_ctx[0], left_ctx[1])
                ).strip()

            right_text = None
            if right_ctx:
                right_text = " ".join(
                    words[j]["text"].strip() for j in range(right_ctx[0], right_ctx[1])
                ).strip()

            print(f"    Block {bi}: words {block_start}–{block_end-1}  "
                  f"({len(raw_text)} chars, {block_end - block_start} words)",
                  flush=True)
            if left_ctx:
                print(f"      left context:  words {left_ctx[0]}–{left_ctx[1]-1}", flush=True)
            if right_ctx:
                print(f"      right context: words {right_ctx[0]}–{right_ctx[1]-1}", flush=True)

            punct_text = self._llm(
                raw_text, left_context=left_text, right_context=right_text)
            if punct_text is None:
                print(f"      LLM failed → native breaks only", flush=True)
                continue

            new_breaks = self._find_new_breaks(
                raw_text, punct_text, block_start, block_words)

            # Programmatic filter: reject breaks where the left side ends
            # with a FRAGILE_TRAILING word (e.g. "going", "the", "and").
            n_accepted = 0
            n_rejected = 0
            for b in new_breaks:
                left_of_break = " ".join(
                    w["text"].strip() for w in words[block_start:b]
                ).strip()
                if self._FRAGILE_RE.search(left_of_break):
                    n_rejected += 1
                    continue
                if would_break_phrase(words, b):
                    n_rejected += 1
                    continue
                """
                Reject LLM-added period before intensifier "so".
                Conjunction "so" (meaning "therefore") is always followed by a clause subject (pronoun/noun).  
                If the word after "so" is NOT a CLAUSE_STARTER, 
                "so" is an intensifier (adverb: "very / that much") and splitting leaves a fragment.
                """
                if b < len(words):
                    wb = words[b]["text"].strip().lower().rstrip('.,!?;\'"')
                    if wb == 'so' and b + 1 < len(words):
                        nxt = words[b + 1]["text"].strip().lower().rstrip('.,!?;\'"')
                        nxt_root = nxt.split("'")[0] if "'" in nxt else nxt
                        if nxt_root not in _CLAUSE_STARTERS:
                            n_rejected += 1
                            continue
                all_breaks.add(b)
                n_accepted += 1

            if new_breaks:
                msg = f"      +{n_accepted} breaks from LLM"
                if n_rejected:
                    msg += f" (rejected {n_rejected} fragile/bigram/so)"
                print(msg, flush=True)
            else:
                print(f"      no new breaks", flush=True)

            # ── Inject LLM-added commas into word-level data ────────────
            # Phase 6/8/10 all benefit from richer comma information.
            # These commas persist in words[i]["text"] for the rest of the
            # pipeline (Phase 5+).
            new_commas = self._find_new_commas(
                raw_text, punct_text, block_start, block_words)
            if new_commas:
                for word_idx in sorted(new_commas):
                    w = words[word_idx]
                    if not w["text"].rstrip().endswith(','):
                        w["text"] = w["text"].rstrip() + ","
                print(f"      +{len(new_commas)} commas injected into words",
                      flush=True)

        # ── Phase 5: Build segments from ALL breaks ───────────────────────
        sorted_breaks = sorted(all_breaks)
        if not sorted_breaks or sorted_breaks[-1] < len(words):
            sorted_breaks.append(len(words))

        # Track word-index ranges alongside segment dicts for force-split
        segments_with_idx = []  # (word_start, word_end, segment_dict)
        prev = 0
        for b in sorted_breaks:
            if b <= prev:
                continue
            seg_words = words[prev:b]
            text = " ".join(w["text"].strip() for w in seg_words).strip()
            if text:
                segments_with_idx.append((
                    prev, b,
                    {
                        "text": text,
                        "start": seg_words[0]["start"],
                        "end": seg_words[-1]["end"],
                    },
                ))
            prev = b

        print(f"    Result: {len(segments_with_idx)} segments", flush=True)

        """
        Phase 6 & 7: Force-split remaining overlength segments.

        Phase 6 (comma split) — splits at clause-internal commas (comma followed by pronoun, conjunction, WH-word, etc.).  
        List commas ("a, b and c") are NOT split.  
        Right-to-left single-split recursion.

        Phase 7 (conjunction split) — for segments that survive Phase 6.
        Splits at coordinating conjunctions (and/but/or/so) that introduce new clauses.  
        Uses rules for but/so/or/and+CLAUSE_STARTER, 
        and optionally the LLM for ambiguous "and" cases.
        """
        n_split_6 = 0
        n_split_7 = 0
        final = []
        for ws, we, seg in segments_with_idx:
            seg_words = words[ws:we]
            if len(seg_words) < 2:
                final.append(seg)
                continue

            seg_text_preview = ' '.join(w['text'] for w in seg_words[:6])
            print(f"  Segment (ws={ws}, we={we}, words={len(seg_words)}): "
                  f"\"{seg_text_preview}...\"", flush=True)

            # Phase 6: Comma split
            sub = _comma_split(seg_words, min_words=min_words)
            if len(sub) > 1:
                for sub_words in sub:
                    sub_text = ' '.join(w['text'] for w in sub_words).strip()
                    if not sub_text:
                        continue

                    # Phase 7: Conjunction split — sub-segments that are
                    # STILL overlength after Phase 6 also get a Phase 7 pass.
                    sub_dur = sub_words[-1]['end'] - sub_words[0]['start']
                    sub_chars = sum(len(w['text']) + 1 for w in sub_words)
                    if (len(sub_words) > max_words or sub_chars > max_chars
                            or sub_dur > max_dur):
                        ambiguous = _find_ambiguous_conjunctions(sub_words, min_words)
                        llm_confirmed_ids = set()
                        if ambiguous:
                            text = ' '.join(w['text'] for w in sub_words)
                            positions_1based = [p + 1 for p in ambiguous]
                            llm_results = self._classify_conjunctions(
                                text, positions_1based, sub_words)
                            if llm_results:
                                llm_confirmed_ids = {
                                    id(sub_words[ambiguous[i]])
                                    for i, yes in enumerate(llm_results)
                                    if yes
                                }
                            print(f"    Phase 7 LLM result: pos={positions_1based} "
                                  f"results={llm_results} seg=\"{text[:80]}...\"",
                                  flush=True)
                        # Pre-scan: identify list conjunctions using full
                        # right_words BEFORE recursion truncates them.
                        list_conjunctions = set()
                        for iw, w in enumerate(sub_words):
                            wt = w["text"].rstrip(',.').lower()
                            if wt not in ('and', 'or'):
                                continue
                            right_words = [
                                sub_words[j]["text"].strip().lower().rstrip('.,!?;\'"')
                                for j in range(iw + 1, len(sub_words))
                            ]
                            if right_words and _is_list_comma(right_words):
                                list_conjunctions.add(id(w))
                        sub7 = _conjunction_split(sub_words, min_words,
                                                   llm_confirmed_ids,
                                                   list_conjunctions)
                        if len(sub7) > 1:
                            for sub7_words in sub7:
                                sub7_text = ' '.join(
                                    w['text'] for w in sub7_words).strip()
                                if sub7_text:
                                    final.append({
                                        'text': sub7_text,
                                        'start': sub7_words[0]['start'],
                                        'end': sub7_words[-1]['end'],
                                    })
                            n_split_7 += 1
                        else:
                            final.append({
                                'text': sub_text,
                                'start': sub_words[0]['start'],
                                'end': sub_words[-1]['end'],
                            })
                    else:
                        final.append({
                            'text': sub_text,
                            'start': sub_words[0]['start'],
                            'end': sub_words[-1]['end'],
                        })
                n_split_6 += 1
                continue

            # Phase 7: Conjunction split — segments that Phase 6 could not
            # split at all but are STILL overlength.
            seg_dur = seg_words[-1]['end'] - seg_words[0]['start']
            seg_chars = sum(len(w['text']) + 1 for w in seg_words)
            if (len(seg_words) > max_words or seg_chars > max_chars
                    or seg_dur > max_dur):
                ambiguous = _find_ambiguous_conjunctions(seg_words, min_words)
                llm_confirmed_ids = set()
                if ambiguous:
                    text = ' '.join(w['text'] for w in seg_words)
                    positions_1based = [p + 1 for p in ambiguous]
                    llm_results = self._classify_conjunctions(text, positions_1based, seg_words)
                    if llm_results:
                        llm_confirmed_ids = {
                            id(seg_words[ambiguous[i]])
                            for i, yes in enumerate(llm_results) if yes
                        }

                # Pre-scan: identify list conjunctions using full
                # right_words BEFORE recursion truncates them.
                list_conjunctions = set()
                for iw, w in enumerate(seg_words):
                    wt = w["text"].rstrip(',.').lower()
                    if wt not in ('and', 'or'):
                        continue
                    right_words = [
                        seg_words[j]["text"].strip().lower().rstrip('.,!?;\'"')
                        for j in range(iw + 1, len(seg_words))
                    ]
                    if right_words and _is_list_comma(right_words):
                        list_conjunctions.add(id(w))

                sub = _conjunction_split(seg_words, min_words, llm_confirmed_ids,
                                         list_conjunctions)
                if len(sub) > 1:
                    for sub_words in sub:
                        sub_text = ' '.join(w['text'] for w in sub_words).strip()
                        if sub_text:
                            final.append({
                                'text': sub_text,
                                'start': sub_words[0]['start'],
                                'end': sub_words[-1]['end'],
                            })
                    n_split_7 += 1
                else:
                    final.append(seg)
            else:
                final.append(seg)

        if n_split_6:
            print(f"  Comma-split {n_split_6} overlength segments", flush=True)
        if n_split_7:
            print(f"  Conjunction-split {n_split_7} overlength segments", flush=True)


        # ── Phase 8: LLM re-punctuation for remaining overlength segments ──
        """
        Segments that survived Phase 6 (comma) and Phase 7 (conjunction)
        but are still overlength are likely run-on sentences 
        where the initial LLM pass didn't add enough periods.  
        Send them to the LLM with a focused "break up run-on sentences" prompt, 
        then apply the same protection chain before splitting.

        New protection:
          Conjunction fragment check 
          — if a proposed break would leave <=4 words on the left starting with and/but/so/or, reject it.
          (This prevents Phase 8 from creating parasitic conjunction fragments that are too short to stand alone.)
        """
        _PHASE8_CONJ_FRAG = frozenset({'and', 'but', 'so', 'or'})
        _PHASE8_PROMPT = (
            "Fix punctuation in the ASR transcript below. This segment "
            "is too long for a subtitle — add periods (. ) to break it "
            "into shorter, readable sentences.\n\n"
            "IMPORTANT: Break up long chains connected by 'and' / 'so' / "
            "'and then'. Whenever you see a complete thought (has a "
            "subject and verb), add a period to separate it.\n\n"
            "IMPORTANT: Add a period BEFORE 'and' / 'but' / 'so' / 'or' "
            "when they start a new clause. Good: \"...we can do that. "
            "And then we move on...\" Not: \"...we can do that and then.\""
            "\n\n"
            "NEVER add a period after: just, almost, nearly, really, "
            "barely, hardly, merely, simply, quite, very, too, rather, "
            "only, still, already, even, also, not, no, some, any.\n\n"
            "Never change wording. Never add explanations. Output ONLY "
            "the punctuated text."
        )

        n_phase8 = 0
        phase8_result = []
        for seg in final:
            seg_word_list = seg["text"].split()
            seg_dur = seg["end"] - seg["start"]
            seg_chars = len(seg["text"])

            if not (len(seg_word_list) > max_words
                    or seg_chars > max_chars
                    or seg_dur > max_dur):
                phase8_result.append(seg)
                continue

            # Locate word range by matching timestamps (tolerance 50 ms)
            ws = None
            for i, w in enumerate(words):
                if abs(w["start"] - seg["start"]) < 0.05:
                    ws = i
                    break
            we = None
            for i in range(len(words) - 1, -1, -1):
                if abs(words[i]["end"] - seg["end"]) < 0.05:
                    we = i + 1
                    break

            if ws is None or we is None or ws >= we or we > len(words):
                phase8_result.append(seg)
                continue

            seg_words = words[ws:we]
            raw_text = " ".join(w["text"].strip() for w in seg_words)

            print(f"  Phase 8: {len(seg_words)}w, {seg_chars}c, "
                  f"{seg_dur:.1f}s  [{seg['text'][:80]!r}]", flush=True)

            punct_text = self._llm(raw_text, max_retries=1,
                                   system_override=_PHASE8_PROMPT)
            if punct_text is None:
                print(f"    LLM failed -> keep original", flush=True)
                phase8_result.append(seg)
                continue

            new_breaks = self._find_new_breaks(raw_text, punct_text,
                                                ws, seg_words)
            if not new_breaks:
                print(f"    no new breaks from LLM", flush=True)
                phase8_result.append(seg)
                continue

            # ── Phase 8 recursive split ──
            """
            Instead of applying all accepted breaks at once (which risks creating intermediate fragments like "and the subscribe"),
            recursively pick the single best break that minimizes left-right word-count imbalance.  
            This follows the same philosophy as _comma_split: each level picks exactly one break and recurses on both halves.
            """
            def _pick_break(ws, we, candidates):
                """Among *candidates* (sorted global word indices within [ws, we)), 
                pick the one that minimizes |left-right word count| and passes all guards.  
                Returns the break index, or None."""
                best = None
                best_diff = float('inf')
                for b in candidates:
                    if b <= ws or b >= we:
                        continue
                    lc = b - ws
                    rc = we - b
                    if lc < min_words or rc < min_words:
                        continue
                    left_text = " ".join(
                        w["text"].strip() for w in words[ws:b]).strip()
                    # 1. FRAGILE_RE — no break after fragile trailing words
                    if self._FRAGILE_RE.search(left_text):
                        continue
                    # 2. Phrase bigram — don't split inside fixed expressions
                    if would_break_phrase(words, b):
                        continue
                    # 3. Conjunction fragment — left side ≤4 words starting
                    #    with and/but/so/or
                    lw = left_text.split()
                    if lw:
                        fl = lw[0].strip().lower().rstrip('.,!?;:\'"')
                        if len(lw) <= 4 and fl in _PHASE8_CONJ_FRAG:
                            continue
                    # 5. Intensifier so check
                    if b < len(words):
                        wb_t = words[b]["text"].strip().lower().rstrip(
                            '.,!?;\'"')
                        if wb_t == 'so' and b + 1 < len(words):
                            nxt = words[b + 1]["text"].strip().lower().rstrip(
                                '.,!?;\'"')
                            nxt_r = nxt.split("'")[0] if "'" in nxt else nxt
                            if nxt_r not in _CLAUSE_STARTERS:
                                continue
                    """
                    Guard 6 — List enumeration guard: don't break at "and"/"or"
                    when the left side has a comma nearby.  
                    Prevents the LLM from adding a period inside "parameters, buttons, and knobs".

                    Phase 7 already avoids splitting here, 
                    but a 20w/130c segment that skips Phase 7 lands in Phase 8 where the LLM may add a period instead.

                    NOTE: WhisperX does not reliably put commas on every list item.  
                    In "parameters, buttons, and knobs" the comma may only appear on "parameters," not on "buttons".  
                    So instead of checking only the immediate left word, 
                    we scan up to 3 words left for any comma.
                    """
                    if b > ws:
                        wb_t = words[b]["text"].strip().lower().rstrip(
                            '.,!?;\'"')
                        if wb_t in ('and', 'or') and b + 1 < len(words):
                            found_comma_left = any(
                                words[j]["text"].strip().endswith(',')
                                for j in range(max(ws, b - 3), b)
                            )
                            if found_comma_left:
                                right_words = [
                                    words[j]["text"].strip().lower().rstrip(
                                        '.,!?;\'"')
                                    for j in range(b, we)
                                ]
                                if _is_list_comma(right_words):
                                    continue
                    diff = abs(lc - rc)
                    if diff < best_diff:
                        best_diff = diff
                        best = b
                return best

            def _split_recursive(ws, we, all_breaks, depth=0):
                """Recursive single-split.  Pick the single most-balanced break within [ws, we).  
                Split there, then recurse on both halves if they are still overlength."""
                if depth > 8:  # safety cap
                    sub = words[ws:we]
                    text = " ".join(
                        w["text"].strip() for w in sub).strip()
                    if text:
                        return [{"text": text,
                                 "start": sub[0]["start"],
                                 "end": sub[-1]["end"]}]
                    return []

                sub = words[ws:we]
                sub_dur = sub[-1]['end'] - sub[0]['start'] if sub else 0
                sub_chars = sum(len(w['text']) + 1 for w in sub)
                if not (len(sub) > max_words or sub_chars > max_chars
                        or sub_dur > max_dur):
                    text = " ".join(
                        w["text"].strip() for w in sub).strip()
                    if text:
                        return [{"text": text,
                                 "start": sub[0]["start"],
                                 "end": sub[-1]["end"]}]
                    return []

                b = _pick_break(ws, we, all_breaks)
                if b is None:
                    sub = words[ws:we]
                    text = " ".join(
                        w["text"].strip() for w in sub).strip()
                    if text:
                        return [{"text": text,
                                 "start": sub[0]["start"],
                                 "end": sub[-1]["end"]}]
                    return []

                return (_split_recursive(ws, b, all_breaks, depth + 1)
                        + _split_recursive(b, we, all_breaks, depth + 1))

            sub_segments = _split_recursive(ws, we, sorted(new_breaks))
            if len(sub_segments) <= 1:
                print(f"    no valid break -> keep original", flush=True)
                phase8_result.append(seg)
            else:
                phase8_result.extend(sub_segments)
                n_phase8 += 1
                print(f"    recursive split -> {len(sub_segments)} segments",
                      flush=True)

        if n_phase8:
            print(f"  Phase 8 re-punctuated {n_phase8} overlength segments",
                  flush=True)

        # ── Phase 9: LLM-guided merge of conjunction-led fragments ────────
        """
        Segments ≤8 words starting with and/but/so/or are likely parasitic fragments that belong to the preceding segment.  
        Ask the LLM to judge whether each such fragment is a CONTINUATION of the previous thought (merge backward) 
        or a NEW_SENTENCE (keep separate).
        Only merge if the LLM confirms CONTINUATION and the merged result does not exceed size limits.  
        Non-overlength segments also pass through this phase — it is a pure merge pass, not a split pass.
        """
        _CONJ_MERGE_WORDS = frozenset({'and', 'but', 'so', 'or'})
        n_phase9 = 0

        # Collect candidates: (index_in_phase8_result, seg_dict, first_word_lower)
        phase9_candidates = []
        for i, seg in enumerate(phase8_result):
            words_list = seg["text"].split()
            if not words_list:
                continue
            first_lower = words_list[0].strip().lower().rstrip('.,!?;:\'"')
            if not (len(words_list) <= 8
                    and first_lower in _CONJ_MERGE_WORDS
                    and i > 0):
                continue
            """
            Hard rule: if the previous segment ends with .?! it is an independent sentence 
            — never merge across a period boundary.
            This prevents Phase 9 from creating segments with internal periods 
            (which would be wrong for both display and subsequent translation passes).
            """
            prev_text = phase8_result[i - 1]["text"].rstrip()
            if prev_text.endswith(('.', '?', '!')):
                continue
            phase9_candidates.append((i, seg, first_lower))

        if phase9_candidates:
            # Build LLM items: (prev_text, curr_text)
            llm_items = [
                (phase8_result[idx_in_p8 - 1]["text"], seg["text"])
                for idx_in_p8, seg, _ in phase9_candidates
            ]
            results = self._classify_conj_merge(llm_items)

            if results is not None:
                # Determine which indices to skip (merged away)
                skip_indices = set()
                for item_idx, (idx_in_p8, seg, fl) in enumerate(phase9_candidates):
                    if item_idx < len(results) and results[item_idx]:
                        # LLM says CONTINUATION — try merge
                        prev_seg = phase8_result[idx_in_p8 - 1]
                        merged_text = prev_seg["text"] + " " + seg["text"]
                        merged_words = len(merged_text.split())
                        merged_chars = len(merged_text)
                        if merged_words <= max_words and merged_chars <= max_chars:
                            skip_indices.add(idx_in_p8)

                # Build output
                phase9_result = []
                for i, seg in enumerate(phase8_result):
                    if i in skip_indices:
                        # Consumed by merge into previous segment
                        prev_seg = phase9_result[-1]
                        phase9_result[-1] = {
                            "text": prev_seg["text"] + " " + seg["text"],
                            "start": prev_seg["start"],
                            "end": seg["end"],
                        }
                        n_phase9 += 1
                    else:
                        phase9_result.append(dict(seg))

                if n_phase9:
                    print(f"  Phase 9 merged {n_phase9} conjunction-led "
                          f"fragments", flush=True)
            else:
                # LLM failed — pass through unchanged
                phase9_result = [dict(seg) for seg in phase8_result]
        else:
            # No candidates — pass through unchanged
            phase9_result = [dict(seg) for seg in phase8_result]

        # ── Phase 10 inner helper ────────────────────────────────────────────
        def _phase10_within_limits(seg_words):
            """Check if a word list is within Phase 10 size limits."""
            n = len(seg_words)
            if not n:
                return True
            chars = sum(len(w['text']) + 1 for w in seg_words)
            return not (n > max_words
                        or (chars > max_chars and n > 20))

        def _phase10_split(seg_words):
            """Recursive Phase 10 split: comma → conjunction → forced mid-split.

            Three rounds with decreasing semantic signal:
              1. Comma split (most balanced, comma word not counted in sides)
              2. Conjunction / subordinator split (most balanced, word goes right)
              3. Forced mid-split (last resort)

            Each split recurses on both halves until all segments are within limits.
            """
            n = len(seg_words)
            if not n or _phase10_within_limits(seg_words):
                return [seg_words]

            # ── Round 1: Comma split (most balanced) ──
            # Scan all commas, pick the most balanced (closest to equal sides).
            # The comma word itself is NOT counted on either side; side lengths
            # must each be >= min_words.
            best_i = None
            best_diff = float('inf')
            for i in range(n - 1, -1, -1):
                w = seg_words[i]
                if not w["text"].rstrip().endswith(','):
                    continue
                left_cnt = i               # words before the comma word
                right_cnt = n - i - 1      # words after the comma word
                if left_cnt < min_words or right_cnt < min_words:
                    continue
                # List-comma check (same logic as _emergency_comma_split)
                right_words = []
                for j in range(i + 1, n):
                    wt = seg_words[j]["text"].strip()
                    right_words.append(wt.lower().rstrip('.,!?;\'"'))
                    if wt.endswith(','):
                        break
                if _is_list_comma(right_words):
                    continue
                diff = abs(left_cnt - right_cnt)
                if diff < best_diff:
                    best_diff = diff
                    best_i = i

            if best_i is not None:
                split_idx = best_i + 1  # comma word stays with left
                result = []
                result.extend(_phase10_split(seg_words[:split_idx]))
                result.extend(_phase10_split(seg_words[split_idx:]))
                return result

            # ── Round 2: Conjunction / subordinator split (most balanced) ──
            # Scan all qualifying conjunctions and subordinators, pick the
            # most balanced.  Split BEFORE the word (word attaches to right).
            best_i = None
            best_diff = float('inf')

            for i in range(n - 1, -1, -1):
                wt = seg_words[i]["text"].rstrip(',.').lower()

                # ── Coordinating conjunctions: and/but/so/or ──
                if wt in ('and', 'but', 'or', 'so'):
                    left_cnt = i           # words before the conjunction
                    right_cnt = n - i      # conjunction + words after
                    if left_cnt < min_words or right_cnt < min_words:
                        continue
                    can_split = False
                    if wt == 'but':
                        can_split = True
                    elif i + 1 < n:
                        first_right = seg_words[i + 1]["text"].strip().lower().rstrip(',.')
                        root = first_right.split("'")[0] if "'" in first_right else first_right
                        if root in _CLAUSE_STARTERS:
                            can_split = True
                    if can_split:
                        diff = abs(left_cnt - right_cnt)
                        if diff < best_diff:
                            best_diff = diff
                            best_i = i

                # ── Subordinators: because/although/since/unless/while/when/where/if/as ──
                elif wt in ('because', 'although', 'since', 'unless',
                            'while', 'when', 'where', 'if', 'as'):
                    left_cnt = i
                    right_cnt = n - i
                    if left_cnt < min_words or right_cnt < min_words:
                        continue
                    # Subordinators always introduce clauses — no list ambiguity
                    diff = abs(left_cnt - right_cnt)
                    if diff < best_diff:
                        best_diff = diff
                        best_i = i

            if best_i is not None:
                result = []
                result.extend(_phase10_split(seg_words[:best_i]))
                result.extend(_phase10_split(seg_words[best_i:]))
                return result

            # ── Round 3: Forced mid-split ──
            mid = n // 2
            result = []
            if mid > 0:
                result.extend(_phase10_split(seg_words[:mid]))
            if mid < n:
                result.extend(_phase10_split(seg_words[mid:]))
            return result

        # ── Phase 10: Emergency split for extreme overlength ─────────
        # Phase 8/9 may leave segments that are still overlength with no good
        # period break.  Phase 10 is the final fallback: comma split (relaxed,
        # most-balanced), conjunction/subordinator split, then forced mid-split.
        n_phase10 = 0
        phase10_result = []
        for seg in phase9_result:
            seg_word_list = seg["text"].split()
            seg_chars = len(seg["text"])

            # Entry: words > 30  OR  (chars > 120 AND words > 20)
            if not (len(seg_word_list) > max_words
                    or (seg_chars > max_chars and len(seg_word_list) > 20)):
                phase10_result.append(seg)
                continue

            # Locate word range by matching timestamps (tolerance 50 ms)
            ws = None
            for i, w in enumerate(words):
                if abs(w["start"] - seg["start"]) < 0.05:
                    ws = i
                    break
            we = None
            for i in range(len(words) - 1, -1, -1):
                if abs(words[i]["end"] - seg["end"]) < 0.05:
                    we = i + 1
                    break
            if ws is None or we is None or ws >= we or we > len(words):
                phase10_result.append(seg)
                continue

            seg_words = words[ws:we]
            sub = _phase10_split(seg_words)

            if len(sub) <= 1:
                phase10_result.append(seg)
            else:
                for sub_words in sub:
                    sub_text = " ".join(
                        w["text"].strip() for w in sub_words).strip()
                    if sub_text:
                        phase10_result.append({
                            "text": sub_text,
                            "start": sub_words[0]["start"],
                            "end": sub_words[-1]["end"],
                        })
                n_phase10 += 1
                print(f"  Phase 10: split -> {len(sub)} segments  "
                      f"[{seg['text'][:60]!r}]", flush=True)

        if n_phase10:
            print(f"  Phase 10 split {n_phase10} still-overlength segments",
                  flush=True)

        return phase10_result


def segment_with_llm(words, model="phi4", model_path=None, server_path=None,
                     port=8088, num_threads=12, context_size=4096,
                     gpu_layers=None, auto_start=True,
                     max_chars=120, max_dur=9.0, max_words=30,
                     min_words=4, min_dur=1.5):
    """Convenience function: create, start, segment, stop, return segments.

    Use this for one-shot CLI invocations.
    """
    pipe = LLMPipeline(
        model=model, model_path=model_path, server_path=server_path,
        port=port, num_threads=num_threads, context_size=context_size,
        gpu_layers=gpu_layers,
    )
    if auto_start:
        if not pipe.start():
            return None
    try:
        return pipe.segment(words, max_chars=max_chars, max_dur=max_dur,
                           max_words=max_words, min_words=min_words, min_dur=min_dur)
    finally:
        if auto_start:
            pipe.stop()
