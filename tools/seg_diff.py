"""
seg_diff.py — Character-diff tools for detecting LLM-added punctuation.

Extracted from :mod:`tools.segment` to reduce bloat. Uses :mod:`difflib`
``SequenceMatcher`` to map LLM output punctuation back to input word indices.

Contents:
  * :func:`_build_char_to_word` — Map char positions to word indices
  * :func:`_find_new_breaks` — Find word indices where LLM added .?!
  * :func:`_find_new_commas` — Find word indices where LLM added commas
"""

import difflib
import re


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

    char_to_word = _build_char_to_word(raw_text)

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

    char_to_word = _build_char_to_word(raw_text)

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
