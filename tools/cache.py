"""Cache word-level data as JSON for re-split debugging."""

import json


def save_words_cache(all_words, cache_path):
    """Persist extracted word data as JSON."""
    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(all_words, f, ensure_ascii=False, indent=2)


def load_words_cache(cache_path):
    """Load previously cached word data."""
    with open(cache_path, 'r', encoding='utf-8') as f:
        return json.load(f)
