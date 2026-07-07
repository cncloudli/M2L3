# LLM Segmentation Pipeline: 10-Phase Algorithm

## End-to-End Flow

```
ASR Word List (with timestamps)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Phases 1-4  LLM Punctuation Infill (sentence boundary      │
│               detection)                                    │
│  Phase 1: Group by native .?! (rule-based)                  │
│  Phase 2: Mark overlength groups (rule-based)               │
│     ├── No overlength groups ──→ Fast path → final segments │
│     │                           (skip phases 3-10)          │
│     └── Has overlength groups ──→ Phase 3                   │
│  Phase 3: Merge adjacent overlength groups into "blocks"    │
│           (rule-based)                                      │
│  Phase 4: LLM punctuation infill + char diff analysis →     │
│           new breaks (LLM)                                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 5: Build segments from all breaks (rule-based)       │  
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────┐
           │ Check overlength?   │──── No ──→ Phase 9 (merge)
           │(words>30 / chars>120│
           │  / dur>9s)          │
           └─────────┬───────────┘
                     │ Yes
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 6: Comma forced split (rule-based)                   │       
│  Split at clause commas, skip list commas                   │   
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────┐
           │ Check still overlng?│──── No ──→ Phase 9 (merge)
           └─────────┬───────────┘
                     │ Yes
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 7: Conjunction split (rule-based + optional LLM)     │  
│  Rule layer → ambiguous and/or → LLM binary classifier      │ 
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────┐
           │ Check still overlng?│──── No ──→ Phase 9 (merge)
           └─────────┬───────────┘
                     │ Yes
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 8: LLM run-on repunctuation — recursive split (LLM)  │ 
│  Phase-8 prompt → recursive single-split → guard checks     │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 9: LLM-guided conjunction-fragment merge (LLM)       │
│  LLM classifies CONTINUATION (merge) or NEW_SENTENCE (keep) │
│  Hard rule: preceding segment ends with .?! → never merge   │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────┐
           │ Check still overlng?│──── No ──→ Output final segs
           │ (words>30 OR        │
           │  chars>120 AND      │
           │  words>20)          │
           └─────────┬───────────┘
                     │ Yes
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  Phase 10: Emergency split — last resort (rule-based)       │
│  Round 1: Comma → Round 2: Conjunction → Round 3: Midpoint  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
                    Final segment list
```

---

## Phase Details

### Phase 1: Group by Native .?!

| Field | Description |
|-------|-------------|
| **Purpose** | Use WhisperX's existing sentence-ending punctuation (.?!) to split the word list into natural sentence units |
| **Input** | Word list (`words`: each word has `{text, start, end}`) |
| **Algorithm** | Iterate through words; whenever a word ends with `.`/`?`/`!`, cut a group (`(start_idx, end_idx)` range) |
| **Output** | `groups`: list of word-index ranges, each ending with native .?! |
| **Function** | [`_build_groups()`](../tools/llm_pipeline.py) |
| **Edge Cases** | Groups without punctuation are preserved as a single group |

```
Example:
Input words: ["Hello", "world.", "This", "is", "great!"]
Output groups: [(0,2), (2,5)]    
               ← Group 1: ["Hello","world."], Group 2: ["This","is","great!"]
```

### Phase 2: Mark Overlength Groups

| Field | Description |
|-------|-------------|
| **Purpose** | Identify groups exceeding size limits — these need further LLM-based splitting |
| **Input** | `groups` (from Phase 1) |
| **Conditions** | A group is marked "long" if any condition is met:<br>• Word count > `max_words` (default 30)<br>• Total chars > `max_chars` (default 120)<br>• Duration > `max_dur` (default 9.0s) |
| **Output** | `long_group_set` (set of overlength group indices), `n_short` (short group count), `n_long` (long group count) |
| **Special** | No overlength groups → fast path: build segments from native breaks directly, skip phases 3–10 |

### Phase 3: Merge Adjacent Overlength Groups into Blocks

| Field | Description |
|-------|-------------|
| **Purpose** | Merge consecutive overlength groups into contiguous "blocks" to reduce LLM calls |
| **Input** | `long_group_set`, `sorted_groups` |
| **Algorithm** | Iterate through sorted groups; consecutive `(gs, ge) ∈ long_group_set` merge into one `block = [(gs,ge), ...]` |
| **Output** | `blocks`: list of blocks, each block is a list of one or more contiguous long group ranges |

```
Diagram:
Groups: [A_short] [B_long] [C_long] [D_short] [E_long] [F_short]
                              ↓
Blocks:          [B+C]                 [E]
          (adjacent long merged)   (isolated long alone)
```

### Phase 4: LLM Punctuation Infill + Diff Analysis

| Field | Description |
|-------|-------------|
| **Purpose** | For each block, call LLM to add missing punctuation, then find new breaks via character diff analysis |
| **Input** | `blocks`, `words`, short groups (as read-only context) |
| **Sub-steps** | ① Find nearest short groups as context (left/right) for each block<br>② Call `_llm()` to send to Phi-4, requesting missing commas and sentence-ending punctuation<br>③ `_find_new_breaks()`: char-level diff of original vs LLM output, identify new .?!<br>④ `_find_new_commas()`: same diff approach to find new commas<br>⑤ **Guard filtering**: reject breaks before fragile trailing words (FRAGILE_RE), inside phrasal bigrams, or before intensifier "so"<br>⑥ Inject new commas back into word data |
| **Output** | `all_breaks`: global set of word indices for breaks (native + newly added) |
| **Functions** | [`_llm()`](../tools/llm_pipeline.py), [`_find_new_breaks()`](../tools/llm_pipeline.py), [`_find_new_commas()`](../tools/llm_pipeline.py), [`_find_context()`](../tools/llm_pipeline.py) |

```
LLM Prompt (default):
  "Fix punctuation in the ASR transcript below. Keep existing punctuation
   that is correct, but ADD missing commas and sentence-ending punctuation
   (. ? !) wherever natural reading requires them..."

Diff Analysis Process:
  Original: "This is great and it works perfectly"
  LLM:      "This is great. And it works perfectly."
                         ↑
                   diff finds new . → new break
  Guard check: ensure left side of break doesn't end with fragile word,
               doesn't break phrasal bigram, isn't intensifier "so"
```

### Phase 5: Build Segments from All Breaks

| Field | Description |
|-------|-------------|
| **Purpose** | Slice the word list at all accumulated break points and construct subtitle segments |
| **Input** | `all_breaks` (all .?! breaks from Phase 4 + native breaks), `words` |
| **Algorithm** | Sort all breaks → slice at each break → build `{text, start, end}` segment dicts |
| **Output** | `segments_with_idx`: list of `(word_start, word_end, segment_dict)` |
| **Function** | Inline in [`segment()`](../tools/llm_pipeline.py) |

### Phase 6: Comma Forced Split

| Field | Description |
|-------|-------------|
| **Purpose** | For segments still overlength, split at clause-internal commas (not list commas) |
| **Input** | `segments_with_idx` (from Phase 5), `words`, `min_words` (default 4) |
| **Algorithm** | Right-to-left scan for qualifying commas:<br>• At least `min_words` words on each side of the comma<br>• First word after comma is `CLAUSE_STARTER` (pronoun/conjunction/WH-word) or `ELABORATION_STARTER` (adverb/comparative/determiner)<br>• Not a list comma (`_is_list_comma()` — has and/or on right with no clause signal)<br>• Pick the rightmost qualifying comma → split → recurse on both sub-segments |
| **Output** | Refined segment list |
| **Functions** | [`_comma_split()`](../tools/llm_pipeline.py), [`_is_list_comma()`](../tools/llm_pipeline.py) |

```
Example:
Input: "We have a low pass and a high pass filter control for the noise and the width
        of these are all connected to the oscillator"

Split at clause comma:
  ✓ "We have a low pass and a high pass filter control for the noise,"
    └── "and the width of these are all connected to the oscillator"
                                      ↑
                    Comma + "and" → clause subject "the" → clause boundary

Skip list comma:
  "parameters, buttons, and knobs" → no split, "and" connects list items
```

**`_CLAUSE_STARTERS` set:** Subject pronouns (I/you/he/she/it/we/they), demonstratives (this/that/these/those/there), coordinating conjunctions (and/but/or/so/nor/yet/for), subordinating conjunctions (if/when/because/although/since/unless/though/while/where/whereas/as/once/until), WH-words (which/who/whom/whose/what/how/why/whether)

**`_ELABORATION_STARTERS` set:** Comparison/similarity (very/similar/much/more), specification (especially/particularly/including/such/like/notably/namely), scope (mostly/mainly/primarily/largely/typically/generally), condition/dependency (depending/based/compared/according/excluding/except/related), modifier adverbs (essentially/specifically)

### Phase 7: Conjunction Split

| Field | Description |
|-------|-------------|
| **Purpose** | For segments still overlength after Phase 6, split at coordinating conjunctions that introduce a new clause |
| **Input** | Phase 6 output, `words`, `min_words` |
| **Algorithm** | Two-layer structure:<br><br>**Layer 1 (rule-based, no LLM call):**<br>• `but` → always splittable<br>• `so` + `CLAUSE_STARTER` → split (conjunction "so")<br>• `so` + adjective/adverb → **do not** split (intensifier "so", e.g. "so good")<br>• `or` + `CLAUSE_STARTER` → split<br>• `and` + `CLAUSE_STARTER` → split<br><br>**Layer 2 (LLM-assisted):**<br>• `and` + non-`CLAUSE_STARTER` → LLM decides if it connects two complete clauses<br>• Dispatched via `_classify_conjunctions()` as a YES/NO binary question |
| **Output** | Refined segment list |
| **Functions** | [`_conjunction_split()`](../tools/llm_pipeline.py), [`_classify_conjunctions()`](../tools/llm_pipeline.py), [`_find_ambiguous_conjunctions()`](../tools/llm_pipeline.py), [`_is_so_intensifier_target()`](../tools/llm_pipeline.py) |

```
Split Decision Tree (for "and" at position i):
                      ┌───────────────────────────────────────┐
                      │  Conjunction is and/so/or/but?        │
                      └──────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
                  but                so/or               and
              → always split     ┌─────────────┐   ┌────────────────────┐
                                 │ Next word is│   │  Next word is      │
                                 │ CLAUSE_     │   │  CLAUSE_STARTER?   │
                                 │ STARTER?    │   ├──────┬──────┬──────┤
                                 ├──────┬──────┤   │ Yes  │ No   │ Can't│
                                 │ Yes  │ No   │   │ Rule │ LLM  │ Tell │
                                 ├──────┼──────┤   │ split│class.│→don't│
                                 │ Split│Is it │   │      │      │ split│
                                 │      │ Intn-│   └──────┴──┬───┴──────┘
                                 │      │sifier│             │
                                 │      │"so"? │             ▼
                                 │      ├──┬───┤          YES/NO
                                 │    │ Y  │ N │
                                 │    ├────┼───┤
                                 │    │No  │LLM│
                                 │    │    │Cls│
                                 │    │Splt│   │ 
                                 └────┴────┴───┘
```

### Phase 8: LLM Run-on Repunctuation

| Field | Description |
|-------|-------------|
| **Purpose** | For segments that Phase 6+7 could not split, re-send to LLM with a dedicated "split long sentences" prompt |
| **Input** | Segments still overlength after Phase 7, `words` |
| **Prompt** | [`_PHASE8_PROMPT`](../tools/llm_pipeline.py): specifically asks to split complete thoughts joined by "and/so/and then", adding periods before sentence-initial conjunctions |
| **Algorithm** | ① Call `_llm()` with the dedicated `_PHASE8_PROMPT`<br>② `_find_new_breaks()` to identify new .?! break points<br>③ Recursive single-split: `_pick_break()` selects the most balanced break (minimizing |left-right| word count diff) that passes all guards<br>④ `_split_recursive()` splits both sides recursively until all sub-segments fit or no viable break remains<br><br>**Guards (`_pick_break`):**<br>• FRAGILE_RE: left side of break must not end with a fragile word<br>• Phrasal bigrams: must not break fixed expressions<br>• Conjunction fragment: left side ≤4 words starting with and/but/so/or → rejected<br>• Intensifier "so" check<br>• List enumeration guard: break before and/or with a comma on the left → likely a list, don't split |
| **Output** | Refined segment list |
| **Functions** | [`_pick_break()`](../tools/llm_pipeline.py), [`_split_recursive()`](../tools/llm_pipeline.py) |

```
Difference from Phase 4:
  Phase 4: Generic punctuation-fix prompt
  Phase 8: Dedicated "split long sentences" prompt, specifically
           targets and/so/and then concatenation

Recursive Split Process:
  Segment [ws, we) still overlength
      │
      ▼
  Find all candidate break points
      │
      ▼
  Pick most balanced (minimize |left-right| word diff) + passes all guards
      │
      ├──→ Split into [ws, b) + [b, we)
      │                    │
      │            ┌───────┴───────┐
      │            ▼               ▼
      │      Recurse left half  Recurse right half
      │         (max 8 levels deep)
      │
  If no viable break → keep original segment
```

### Phase 9: LLM-Guided Conjunction Fragment Merge

| Field | Description |
|-------|-------------|
| **Purpose** | Merge overly short conjunction-headed "parasitic fragments" back into the preceding segment |
| **Input** | All segments from Phase 8 |
| **Conditions** | Candidate fragment: word count ≤ 8 AND starts with `and/but/so/or`, AND preceding segment does not end with .?! |
| **Algorithm** | ① Collect all candidate fragments `(idx, seg, first_word)`<br>② Call `_classify_conj_merge()`: LLM judges each candidate as CONTINUATION (merge) or NEW_SENTENCE (keep)<br>③ Only merge when LLM confirms CONTINUATION AND the merged result fits within limits |
| **Output** | Refined segment list |
| **Function** | [`_classify_conj_merge()`](../tools/llm_pipeline.py) |

```
Example:
  Preceding segment: "So we have a low pass and a high pass filter"
  Current segment:   "and the width of these are all connected"
                                  ↓
  LLM judge: CONTINUATION
                                  ↓
  After merge: "So we have a low pass and a high pass filter and
                the width of these are all connected"
  (Only if word/char count stays within limits)

Hard rule: Preceding segment ends with .?! → never merge (independent sentence boundary)
```

### Phase 10: Emergency Split

| Field | Description |
|-------|-------------|
| **Purpose** | Last line of defense: force-split any extreme overlength segments that previous phases couldn't handle, without LLM |
| **Input** | Segments still overlength after Phase 9, `words`, `min_words` |
| **Conditions** | Word count > `max_words` (30) OR (char count > `max_chars` (120) AND word count > 20) |
| **Algorithm** | Three rounds:<br><br>**Round 1: Comma split**<br>• Scan all commas, pick the most balanced (minimize |left-right|) that passes list-comma check<br>• Comma word does not count toward either side's word count<br><br>**Round 2: Conjunction/subordinator split**<br>• Coordinating conjunctions (and/but/so/or) + clause subject → splittable, pick most balanced<br>• Subordinators (because/although/since/unless/while/when/where/if/as) → always splittable, pick most balanced<br>• so/or/and without clause subject → don't split here (reserved for LLM phases)<br><br>**Round 3: Force midpoint split**<br>• `mid = n // 2`, split at the exact midpoint<br>• Recurse on both sides until all segments fit limits |
| **Output** | Final segment list (end of the segmentation pipeline) |
| **Functions** | [`_phase10_split()`](../tools/llm_pipeline.py), [`_phase10_within_limits()`](../tools/llm_pipeline.py) |

## LLM Call Summary

| Phase | LLM Call | Prompt | Temp | Purpose |
|-------|----------|--------|------|---------|
| 4 | ✅ Once per block | Generic punctuation fix (with context) | 0 | Add missing sentence-ending punctuation and commas |
| 7 | ⚠️ Only for ambiguous and/or | "YES/NO" binary classification | 0 | Decide if conjunction connects two clauses |
| 8 | ✅ Once per overlength segment | Dedicated run-on split prompt | 0 | Force-split run-on sentences |
| 9 | ✅ Once per batch of candidates | "CONTINUATION/NEW_SENTENCE" binary | 0 | Decide if conjunction fragment should merge back |

> **Note**: Phase 7 and 9 LLM calls are **conditional** — only triggered when ambiguous conjunctions or fragments exist. Phases 4 and 8 always call the LLM.

## Key Guard Mechanisms

| Guard Name | Affected Phases | Purpose |
|------------|----------------|---------|
| **FRAGILE_RE** | 4, 8 | Reject break after fragile trailing words (articles, prepositions, auxiliary verbs, modals, etc.) |
| **Phrasal Bigrams** | 4, 8 | Don't break ~5,700 fixed expressions (e.g. "such as", "going to", "just so") |
| **List Comma Check** | 6, 10 | Don't split at list commas ("a, b and c") |
| **Intensifier "so"** | 4, 7, 8 | Don't split before intensifier "so" ("so good" — adverb of degree); only split before conjunction "so" ("so I" — causal) |
| **Conjunction Fragment Guard** | 8 | Reject creating parasitic fragments ≤4 words starting with a conjunction |
| **List Enumeration and/or** | 7, 8 | Don't split before and/or in list enumerations |
| **Hard Boundary (.?!)** | 9 | Preceding segment ends with .?! → never merge across sentence boundary |
