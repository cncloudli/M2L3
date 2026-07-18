"""
seg_rules.py — Pure rule-based segmentation helpers (no LLM dependency).

Extracted from :mod:`tools.segment` to reduce bloat. All functions are
standalone rule logic — no LLM calls, no file I/O, no class dependencies.

Contents:
  * :func:`_comma_split` — Recursive comma-based force-split
  * :func:`_conjunction_split` — Recursive conjunction-based force-split
  * :func:`_find_ambiguous_conjunctions` — Identify conjunctions needing LLM
  * :func:`_is_list_comma` — Detect list-separator commas
  * :func:`_is_so_intensifier_target` — Detect intensifier "so"
  * Constants: ``_SO_INTENSIFIER_ADJS``, ``_CLAUSE_STARTERS``,
    ``_ELABORATION_STARTERS``, ``_CLAUSE_SUBJECTS``, ``_CONJUNCTIONS``
"""

import re

from tools.non_split_bigrams import would_break_phrase


# ═══════════════════════════════════════════════════════════════════════════════
#  so-intensifier detection
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  Clause / elaboration starters
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  List-comma detection
# ═══════════════════════════════════════════════════════════════════════════════

def _is_list_comma(right_words):
    """Check if a comma separates list items via and/or in the right half.

    Scans the right half for the first ``and`` / ``or``.
    If it looks like a list connector (no clause signal nearby),
    returns True → this is a list comma → don't split there.

    **Early-exit guard**: if the very first word after the comma is a subject
    pronoun/demonstrative (``_CLAUSE_SUBJECTS``) — e.g. *"hours,**we're** going
    to..."* — the comma is a clause boundary, not a list comma.

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


# ═══════════════════════════════════════════════════════════════════════════════
#  Comma-based force-split  (Phase 6)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Comma-based force-split — replaces the old _llm_force_split / _best_split
gap-based approach.
Only splits at commas where the following word is a clause-starter
(pronoun, conjunction, WH-word, etc.) — NOT at list commas ("a, b and c").

Uses right-to-left single-split recursion: find the rightmost
qualifying comma, split there, and recurse on both sub-segments.
This guarantees no fragment is shorter than min_words.
"""

def _comma_split(seg_words, min_words=4):
    """Split an overlength segment at clause-internal commas (recursive).

    For each comma in the segment:
      - Left side (not counting the comma word itself) must have >= min_words
      - Right side (not counting the comma word itself) must have >= min_words
      - First word of right side must be a CLAUSE_STARTER or
        ELABORATION_STARTER — NOT a noun, verb, or list item.

    Algorithm (right-to-left, single-split recursion):
      Scan commas from rightmost to leftmost.
      Take the first (rightmost) comma that satisfies all guards and split at it
      — creating exactly TWO sub-segments.
      Then recurse on each sub-segment.

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
        right_words = []
        for j in range(i + 1, n):
            wt = seg_words[j]["text"].strip()
            right_words.append(wt.lower().rstrip('.,!?;\'"'))
            # Truncate at the next comma — a list connector (and/or)
            # should appear before the next comma in the right half.
            if wt.endswith(','):
                break
        is_list = _is_list_comma(right_words)
        print(f"    Phase 6 comma idx={i} word=\"{w['text']}\" "
              f"right_words={right_words} is_list={is_list}", flush=True)
        if is_list:
            continue

        # Found the rightmost qualifying comma — split ONE comma per call,
        # then recurse on each side.
        split_idx = i + 1  # exclusive-end index (comma word stays with left)
        left_part = seg_words[:split_idx]
        right_part = seg_words[split_idx:]

        result = []
        result.extend(_comma_split(left_part, min_words))
        result.extend(_comma_split(right_part, min_words))
        return result

    # No qualifying comma found
    return [seg_words]


# ═══════════════════════════════════════════════════════════════════════════════
#  Conjunction-based force-split  (Phase 7)
# ═══════════════════════════════════════════════════════════════════════════════

"""
Conjunction-based force-split — targets segments that survived Phase 6
(no splittable comma) but are still overlength.
Splits at coordinating conjunctions (and, but, or, so) that introduce new
independent clauses rather than listing items.

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

    Conjunctions followed by CLAUSE_STARTER (e.g. ``so I``, ``or we``, ``and you``)
    are handled by Tier 1 rules.
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

    Split happens BEFORE the conjunction → the conjunction attaches to the right
    sub-segment, which reads more naturally in subtitles.

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
