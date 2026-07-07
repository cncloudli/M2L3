"""Time-formatting utilities for SRT output."""


def format_srt_time(seconds: float) -> str:
    """Convert seconds (float) to SRT timestamp format HH:MM:SS,mmm."""
    if seconds < 0:
        seconds = 0
    ms = int(round((seconds % 1) * 1000))
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d},{ms:03d}"
