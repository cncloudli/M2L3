"""
Sentence segmentation via LLM punctuation restoration.

Uses :mod:`tools.llm_call` to add missing punctuation and split ASR output
into well-formed subtitle segments.

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

import os, re, subprocess, time, atexit
import sys
from pathlib import Path

from tools.non_split_bigrams import would_break_phrase
from tools.llm_call import create_llm_call, clean_llm_output, NO_PROXY
from tools.seg_rules import (
    _comma_split, _conjunction_split, _find_ambiguous_conjunctions,
    _is_list_comma, _CLAUSE_STARTERS,
)
from tools.seg_diff import _find_new_breaks, _find_new_commas

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


# ══════════════════════════════════════════════════════════════════════════════
#  Fragile trailing words
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
#  Group-by-punctuation
# ══════════════════════════════════════════════════════════════════════════════


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


# ══════════════════════════════════════════════════════════════════════════════
#  Block splitting — prevent overlength blocks
# ══════════════════════════════════════════════════════════════════════════════


def _split_block(block_groups, words, max_block_words, max_block_chars):
    """Recursively split a block at group boundaries if it exceeds limits.

    A block is a list of consecutive long-group ranges ``[(start,end), ...]``.
    Splitting only happens at existing group boundaries — sentences are never
    cut mid-clause.

    Even-sized blocks split in half.  Odd-sized blocks try every split
    point and pick the one that minimises ``max(left_words, right_words)``.
    Single-group blocks that still exceed limits are left as-is (can't split
    without breaking a sentence).

    Returns:
        list of sub-block group lists.
    """
    total_words = sum(ge - gs for gs, ge in block_groups)
    total_chars = sum(
        len(words[j]["text"]) + 1
        for gs, ge in block_groups
        for j in range(gs, ge)
    )

    if total_words <= max_block_words and total_chars <= max_block_chars:
        return [block_groups]

    # Single group — cannot split further
    if len(block_groups) <= 1:
        return [block_groups]

    # Find the split point that minimises max(left_words, right_words)
    n = len(block_groups)
    best_mid = 1
    best_max = float("inf")
    for mid in range(1, n):
        left_words = sum(ge - gs for gs, ge in block_groups[:mid])
        right_words = sum(ge - gs for gs, ge in block_groups[mid:])
        this_max = left_words if left_words > right_words else right_words
        if this_max < best_max:
            best_max = this_max
            best_mid = mid

    result = []
    for sub in (block_groups[:best_mid], block_groups[best_mid:]):
        result.extend(_split_block(sub, words, max_block_words, max_block_chars))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Context resolution for long-group blocks
# ══════════════════════════════════════════════════════════════════════════════


def _find_context(block_groups, sorted_groups, long_group_set,
                  max_context_words=None):
    """Find the immediately adjacent groups before / after *block_groups*.

    If the adjacent group is a **short** group (not in *long_group_set*),
    it is returned whole — a complete sentence works best as context.

    If the adjacent group is a **long** group, it is truncated to
    *max_context_words* from the edge (``…N words before the block`` /
    ``first N words after the block``).  This keeps context focused
    on the boundary rather than overwhelming the LLM with far-away content.

    For unsplit blocks the adjacent groups are short groups (blocks are
    built from consecutive long groups separated by short groups), so this
    behaves identically to the original logic.  For split sub-blocks
    the inner sides fall back to truncated long groups from sibling
    sub-blocks, giving the LLM the information it needs about *what comes
    next* without the full content of the next sub-block.

    Args:
        block_groups: list of ``(start, end)`` for consecutive long groups.
        sorted_groups: sorted list of ALL ``(start, end)`` group ranges.
        long_group_set: set of ``(start, end)`` for overlength groups.
        max_context_words: max words to take from a long-group fallback
            (should match the short-group word limit).

    Returns:
        ``(left_start, left_end) or None``, ``(right_start, right_end) or None``.
    """
    block_start = block_groups[0][0]
    block_end = block_groups[-1][1]

    # Nearest group before the block (any type — short or long).
    left_adj = None
    # Nearest group after the block (any type — short or long).
    right_adj = None

    for gs, ge in sorted_groups:
        if ge <= block_start:
            left_adj = (gs, ge)          # keeps overwriting → closest
        if gs >= block_end and right_adj is None:
            right_adj = (gs, ge)

    left_ctx = None
    if left_adj is not None:
        if (left_adj[0], left_adj[1]) not in long_group_set:
            left_ctx = left_adj           # short group → use whole
        elif max_context_words:
            ctx_start = max(left_adj[1] - max_context_words, left_adj[0])
            left_ctx = (ctx_start, left_adj[1])

    right_ctx = None
    if right_adj is not None:
        if (right_adj[0], right_adj[1]) not in long_group_set:
            right_ctx = right_adj          # short group → use whole
        elif max_context_words:
            ctx_end = min(right_adj[0] + max_context_words, right_adj[1])
            right_ctx = (right_adj[0], ctx_end)

    return left_ctx, right_ctx


# ══════════════════════════════════════════════════════════════════════════════
#  LLM interaction helpers
# ══════════════════════════════════════════════════════════════════════════════


def _llm(llm_call, text, left_context=None, right_context=None, max_retries=2,
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
        llm_call: an :class:`tools.llm_call.LLMCall` instance.
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

    max_tokens = max(64, len(text) // 4 + 30)
    retry_delays = [5, 15, 30]
    for attempt in range(max(max_retries + 1, 1)):
        response = llm_call.chat(
            system=system, user=text,
            max_tokens=max_tokens, temperature=0, cache_prompt=True,
        )
        if response and response.strip():
            return clean_llm_output(response)
        if attempt < max_retries:
            delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
            reason = "LLM call failed" if response is None else "Empty response"
            print(f"    [WARN] {reason}, retrying in {delay}s "
                  f"({attempt + 1}/{max_retries})...", flush=True)
            time.sleep(delay)
    return None


def _classify_conjunctions(llm_call, text, con_positions, seg_words):
    """Ask the LLM which conjunction positions connect two complete clauses.

    Works for ``and``, ``so``, and ``or``.
    Each position is shown with the actual word
    so the LLM can decide based on context
    whether it introduces a new independent clause (YES) or connects list items / modifies an adverb (NO).

    Uses a simple yes/no prompt per batch of positions within one text.
    At temperature=0 the LLM should output a clean comma-separated list of YES/NO tokens.

    Args:
        llm_call: an :class:`tools.llm_call.LLMCall` instance.
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
    max_tokens = max(24, len(con_positions) * 8)
    retry_delays = [5, 15, 30]
    for attempt in range(3):
        response = llm_call.chat(
            system=system, user=user,
            max_tokens=max_tokens, temperature=0, cache_prompt=True,
        )
        if response and response.strip():
            raw = response
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
            # Model may repeat responses (e.g. "NO,NO,NO,NO," for one position).
            # Truncate extra results or pad missing ones.
            if len(results) > len(con_positions):
                return results[:len(con_positions)]
            if len(results) > 0:
                results.extend([False] * (len(con_positions) - len(results)))
                return results
        if attempt < len(retry_delays) - 1:
            delay = retry_delays[attempt]
            reason = "LLM call failed" if response is None else "Empty/malformed response"
            print(f"    [WARN] {reason}, retrying in {delay}s "
                  f"({attempt + 1}/3)...", flush=True)
            time.sleep(delay)
    return None


def _classify_conj_merge(llm_call, items):
    """Classify whether conjunction-led segments are continuations.

    For each (prev_text, curr_text) pair,
    determine whether the conjunction-led current segment
    is a CONTINUATION of the previous sentence (should merge backward) or a NEW_SENTENCE (should not).

    Args:
        llm_call: an :class:`tools.llm_call.LLMCall` instance.
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
    max_tokens = max(8, len(items) * 14)
    retry_delays = [5, 15, 30]
    for attempt in range(3):
        response = llm_call.chat(
            system=system, user=user_text,
            max_tokens=max_tokens, temperature=0, cache_prompt=True,
        )
        if response and response.strip():
            raw = response
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
            # Truncate extra repetitions or pad missing ones
            if len(results) > len(items):
                return results[:len(items)]
            if len(results) > 0:
                results.extend([False] * (len(items) - len(results)))
                return results
        else:
            reason = "LLM call failed" if response is None else "Empty/malformed response"
            print(f"    [Phase 9 merge] {reason}, retrying in "
                  f"{retry_delays[attempt]}s ({attempt + 1}/3)...", flush=True)
        if attempt < len(retry_delays) - 1:
            delay = retry_delays[attempt]
            time.sleep(delay)
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Main entry point
# ══════════════════════════════════════════════════════════════════════════════


def segment_words(words, seg_backend="local", seg_model=None,
            api_key="", base_url="", gpu_layers=None,
            max_chars=120, max_dur=9.0, max_words=30,
            min_words=4):
    """Segment words into sentences via LLM punctuation filling.

    Creates an LLM backend internally, starts/stops it, and runs the
    full 10-phase segmentation pipeline.

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

    Note: Word timestamp fixing is done upstream before this function is called.

    Args:
        words: list of word dicts with keys ``text``, ``start``, ``end``.
        seg_backend: LLM backend name (local, deepseek, openai, …).
        seg_model: model name (None → per-backend default).
        api_key: API key for online backends.
        base_url: base URL override for online backends.
        gpu_layers: GPU offload layers (local backend only; None = auto).
        max_chars: maximum characters per segment.
        max_dur: maximum duration in seconds per segment.
        max_words: maximum words per segment.
        min_words: minimum words per split candidate side.

    Returns:
        list of segment dicts with keys ``text``, ``start``, ``end``.
    """
    llm_call = create_llm_call(
        backend=seg_backend, model=seg_model,
        api_key=api_key, base_url=base_url, gpu_layers=gpu_layers,
    )

    # Lazy-start the LLM backend (started after WhisperX + Wav2Vec2
    # alignment to avoid VRAM contention).
    if not llm_call.start():
        print("[ERR] LLM server failed to start — native breaks only",
              flush=True)

    if not words:
        llm_call.stop()
        return []

    print(f"  LLM punctuation segment: {len(words)} words", flush=True)

    try:

        # ── Phase 1: Native groups at .?! ──────────────────────────────────
        groups = _build_groups(words)
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

        # ── Phase 3: Build blocks of consecutive long groups,
        #             then split overlength blocks at group boundaries ──
        sorted_groups = sorted(groups, key=lambda x: x[0])
        raw_blocks = []
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
                raw_blocks.append(block)
            else:
                i += 1

        # Split overlength blocks at group boundaries so each sub-block
        # stays within the LLM's comfortable working range.
        max_block_words = max_words * 3
        max_block_chars = max_chars * 3
        blocks = []
        for raw in raw_blocks:
            blocks.extend(
                _split_block(raw, words, max_block_words, max_block_chars)
            )

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

            # Find read-only context (nearest short groups; fall back to
            # truncated long group when none are available).
            left_ctx, right_ctx = _find_context(
                block_groups, sorted_groups, long_group_set,
                max_context_words=max_words)

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

            punct_text = _llm(
                llm_call, raw_text, left_context=left_text, right_context=right_text)
            if punct_text is None:
                print(f"      LLM failed → native breaks only", flush=True)
                continue

            new_breaks = _find_new_breaks(
                raw_text, punct_text, block_start, block_words)

            # Programmatic filter: reject breaks where the left side ends
            # with a FRAGILE_TRAILING word (e.g. "going", "the", "and").
            n_accepted = 0
            n_rejected = 0
            for b in new_breaks:
                left_of_break = " ".join(
                    w["text"].strip() for w in words[block_start:b]
                ).strip()
                if _FRAGILE_RE.search(left_of_break):
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
            new_commas = _find_new_commas(
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
                            llm_results = _classify_conjunctions(
                                llm_call, text, positions_1based, sub_words)
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
                    llm_results = _classify_conjunctions(llm_call, text, positions_1based, seg_words)
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

            punct_text = _llm(llm_call, raw_text, max_retries=1,
                              system_override=_PHASE8_PROMPT)
            if punct_text is None:
                print(f"    LLM failed -> keep original", flush=True)
                phase8_result.append(seg)
                continue

            new_breaks = _find_new_breaks(raw_text, punct_text,
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
                    if _FRAGILE_RE.search(left_text):
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
            results = _classify_conj_merge(llm_call, llm_items)

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
            return n <= max_words and chars <= max_chars

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
                        elif wt == 'and':
                            # Phase 10 fallback: "and" without CLAUSE_STARTER
                            # is still better than a forced mid-split that could
                            # land after a FRAGILE_TRAILING word like "a" / "the".
                            # min_words guard prevents conjunction fragments.
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
            # Find a split point near the middle that doesn't leave a
            # FRAGILE_TRAILING word on the left (e.g. "a", "the", "with").
            # Scan outward from center to find the safest split.
            mid = n // 2
            best_mid = None
            for offset in range(n):
                for sign in (1, -1):
                    candidate = mid + offset * sign
                    if candidate < 1 or candidate >= n:
                        continue
                    left_text = " ".join(
                        w["text"].strip() for w in seg_words[:candidate]
                    ).strip()
                    if _FRAGILE_RE.search(left_text):
                        continue
                    best_mid = candidate
                    break
                if best_mid is not None:
                    break
            if best_mid is None:
                best_mid = n // 2  # fallback: exact middle
            mid = best_mid
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

            if not (len(seg_word_list) > max_words
                    or seg_chars > max_chars):
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

    finally:
        llm_call.stop()
