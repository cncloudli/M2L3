"""Extract word-level data from WhisperX / stable-whisper results and fix timestamps."""


def extract_words_from_result(result):
    """Pull (start, end, text) word tuples from a result dict or object.

    Handles both WhisperX dict-style output and stable-whisper
    attribute-style output (backwards compatible).
    """
    segments = (
        result
        if isinstance(result, list)
        else (
            result.get('segments', [])
            if isinstance(result, dict)
            else getattr(result, 'segments', [])
        )
    )
    all_words = []
    for seg_idx, seg in enumerate(segments):
        if isinstance(seg, dict):
            if seg.get('no_speech_prob', 0) > 0.60:
                continue
            words = seg.get('words', [])
        else:
            if getattr(seg, 'no_speech_prob', 0) > 0.60:
                continue
            words = getattr(seg, 'words', [])

        for w in words:
            if isinstance(w, dict):
                ws = w.get('start', None)
                we = w.get('end', None)
                wt = (w.get('word', '') or w.get('text', '')).strip()
            else:
                ws = getattr(w, 'start', None)
                we = getattr(w, 'end', None)
                wt = (getattr(w, 'word', '') or getattr(w, 'text', '')).strip()
            if ws is not None and wt:
                all_words.append({
                    'start': ws, 'end': we, 'text': wt,
                    'seg_idx': seg_idx,  # VAD-derived segment boundary
                })
    return all_words


def fix_word_timestamps(words):
    """Fix ASR word timestamps inflated by trailing silence after a word.

    Wav2Vec2 forced alignment sometimes includes silence/pauses 
    at the end of a word's duration (especially before a long pause or breath).  
    This detects outlier word durations relative to the local speaking rate 
    and clips the trailing silence, leaving the inter-word gap intact.

    Only words far above the local median are modified 
    — typical word-to-word variation is preserved.

    Operates in-place on the list of dicts, returns the list for chaining.
    """
    n = len(words)
    if n < 3:
        return words

    # Global median as fallback
    durs = sorted([w["end"] - w["start"] for w in words if w["end"] > w["start"]])
    global_median = durs[len(durs) // 2] if durs else 0.180

    # Pass 1: clip end times on outliers
    for i in range(n):
        dur = words[i]["end"] - words[i]["start"]

        # Local median (10-word window on each side, exclude self)
        lo = max(0, i - 10)
        hi = min(n, i + 11)
        local = sorted([
            words[j]["end"] - words[j]["start"]
            for j in range(lo, hi)
            if j != i and words[j]["end"] > words[j]["start"]
        ])
        local_median = local[len(local) // 2] if local else global_median

        # Threshold: 5x local median, floor 1.0s
        threshold = max(local_median * 5, 1.0)
        if dur <= threshold:
            continue

        # Cap to 3x local median, floor 0.5s
        cap_dur = max(local_median * 3, 0.5)
        new_end = words[i]["start"] + cap_dur

        # Don't overflow into the next word
        if i + 1 < n:
            new_end = min(new_end, words[i + 1]["start"])

        if new_end < words[i]["end"]:
            words[i]["end"] = new_end

    # Pass 2: resolve overlaps (word ends after next word starts)
    for i in range(n - 1):
        if words[i]["end"] > words[i + 1]["start"]:
            words[i]["end"] = words[i + 1]["start"]

    return words
