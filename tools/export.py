"""Export subtitle data to SRT and TXT formats."""

import os
from tools.format import format_srt_time


def export_srt(result, output_path: str):
    """Write segments to a SRT file (with index + timestamps)."""
    segments = (
        result.get('segments', [])
        if isinstance(result, dict)
        else getattr(result, 'segments', [])
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, seg in enumerate(segments, start=1):
            if isinstance(seg, dict):
                text = seg.get('text', '').strip()
                start = seg.get('start')
                end = seg.get('end')
            else:
                text = getattr(seg, 'text', '').strip()
                start = getattr(seg, 'start', None)
                end = getattr(seg, 'end', None)
            if not text or start is None or end is None:
                continue
            f.write(f"{idx}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")


def export_txt(result, output_path: str, pure_text: bool = True):
    """Write segments to a TXT file.

    *pure_text=True*  → one segment per line (plain text, no timestamps).
    *pure_text=False* → full SRT format (index + timestamp + text).
    """
    segments = (
        result.get('segments', [])
        if isinstance(result, dict)
        else getattr(result, 'segments', [])
    )
    with open(output_path, "w", encoding="utf-8") as f:
        if pure_text:
            for seg in segments:
                text = (
                    seg.get('text', '').strip()
                    if isinstance(seg, dict)
                    else getattr(seg, 'text', '').strip()
                )
                if text:
                    f.write(f"{text}\n")
        else:
            for idx, seg in enumerate(segments, start=1):
                if isinstance(seg, dict):
                    text = seg.get('text', '').strip()
                    start = seg.get('start')
                    end = seg.get('end')
                else:
                    text = getattr(seg, 'text', '').strip()
                    start = getattr(seg, 'start', None)
                    end = getattr(seg, 'end', None)
                if not text or start is None or end is None:
                    continue
                f.write(f"{idx}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")


# def export_word_level(words, output_stem, output_dir):
#     """Export word-level SRT + TXT for debugging.
#
#     Writes ``{output_dir}/{output_stem}_wl.srt`` and ``_wl.txt``
#     with one entry per word (its own timestamp + its text).
#
#     This is a **debug-only** facility — remove from production pipeline.
#     """
#     srt_path = os.path.join(output_dir, f"{output_stem}_wl.srt")
#     txt_path = os.path.join(output_dir, f"{output_stem}_wl.txt")
#
#     with open(srt_path, "w", encoding="utf-8") as f_srt, \
#          open(txt_path, "w", encoding="utf-8") as f_txt:
#         for idx, w in enumerate(words, start=1):
#             text = w["text"].strip()
#             start = w.get("start", 0)
#             end = w.get("end", start)
#             f_srt.write(f"{idx}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}\n\n")
#             f_txt.write(f"{idx:>6d}  {format_srt_time(start)} --> {format_srt_time(end)}  {text}\n")
#
#     print(f"[DBG] Word-level SRT: {srt_path} ({len(words)} words)")
#     print(f"[DBG] Word-level TXT: {txt_path}")
