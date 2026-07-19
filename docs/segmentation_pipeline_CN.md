# LLM 切分流水线：10 阶段算法

## 整体流程

```
ASR 单词列表（带时间戳）
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 1-4  LLM 标点填充（识别句子边界）                       
│                                                             
│  阶段 1: 按原生 .?! 分组 (规则)                              
│  阶段 2: 标记超长组 (规则)                                   
│     ├── 无超长组 ──→ 快速路径 → 最终段（跳过阶段 3-10）
│     └── 有超长组 ──→ 阶段 3
│  阶段 3: 合并连续超长组为"块" (规则)                          
│  阶段 4: LLM 填充缺失标点 + 字符差异分析 → 新增断点 (LLM)    
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 5: 根据所有断点构建字幕段 (规则)                         
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────┐
           │ 检查段是否超长        │──── 否 ──→ 阶段 9（合并检查）
           │ (words>30 / chars>120│
           │  / dur>9s)           │
           └─────────┬───────────┘
                     │ 是
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 6: 逗号强制切分 (规则)                                  
│  从句逗号处切分，跳过列表逗号                                  
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────┐
           │ 检查子段是否仍超长    │──── 否 ──→ 阶段 9（合并检查）
           └─────────┬───────────┘
                     │ 是
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 7: 连词切分 (规则 + 可选 LLM)                           
│  规则层 → 歧义 and/or → LLM 二分类                            
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────┐
           │ 检查段是否仍超长      │──── 否 ──→ 阶段 9（合并检查）
           └─────────┬───────────┘
                     │ 是
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 8: LLM 超长句重标点—递归切分 (LLM)                       
│  专用提示词 → 递归单切分 → 多保护检查                         
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 9: LLM 引导的连词片段合并 (LLM)                          
│  LLM 判断 CONTINUATION（合并）或 NEW_SENTENCE（保留）         
│  硬规则：前段以 .?! 结尾 → 永不合并                            
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────┐
           │ 检查段是否仍超长      │──── 否 ──→ 输出最终段
           │ (words>30 OR         │
           │  chars>120)          │
           └─────────┬───────────┘
                     │ 是
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  阶段 10: 紧急切分—最终兜底 (规则)                             
│  第 1 轮：逗号 → 第 2 轮：连词 → 第 3 轮：强制中点            
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
                    最终段列表
```

---

## 各阶段详细说明

### 阶段 1：按原生 .?! 分组

| 项目 | 内容 |
|------|------|
| **目的** | 利用 WhisperX 已有的句末标点（.?!）将单词列表切分为自然句单位 |
| **输入** | 单词列表（`words`：每个词有 `{text, start, end}`） |
| **算法** | 遍历单词；遇到以 `.`/`?`/`!` 结尾的词即切分一个组（`(start_idx, end_idx)` 范围） |
| **输出** | `groups`：单词索引范围列表，每组以原生 .?! 结尾 |
| **函数** | [`_build_groups()`](../tools/segment.py) |
| **边界情况** | 无标点的组被整体保留为一个组 |

```
示例：
输入词: ["Hello", "world.", "This", "is", "great!"]
输出组: [(0,2), (2,5)]    ← 第1组: ["Hello","world."], 第2组: ["This","is","great!"]
```

### 阶段 2：标记超长组

| 项目 | 内容 |
|------|------|
| **目的** | 识别超过大小限制的组，这些组需要 LLM 进一步切分 |
| **输入** | `groups`（来自阶段 1） |
| **条件** | 满足任一条件即标记为"长组"：<br>• 词数 > `max_words`（默认 30）<br>• 总字符 > `max_chars`（默认 120）<br>• 时长 > `max_dur`（默认 9.0s） |
| **输出** | `long_group_set`（超长组的集合），`n_short`（短组数量），`n_long`（长组数量） |
| **特殊** | 若无超长组 → 快速路径：直接用原生断点构建段，跳过阶段 3-10 |

### 阶段 3：合并连续超长组为块并拆分超大块

| 项目 | 内容 |
|------|------|
| **目的** | 将相邻的超长组合并为连续的"块"（减少 LLM 调用次数），同时将超大的块在组边界处拆分，防止块过大导致 LLM 处理效果下降 |
| **输入** | `long_group_set`，`sorted_groups`，`words` |
| **算法** | 两步：<br><br>**① 合并连续长组**：遍历排序后的组，连续的 `(gs, ge) ∈ long_group_set` 合并为一个 `block = [(gs,ge), ...]`<br><br>**② 递归拆分超大块**：对每个合并后的 raw block，若总词数超过 `max_block_words`（`max_words × 3`，默认 90）或总字符数超过 `max_block_chars`（`max_chars × 3`，默认 360），则调用 `_split_block()` 在**组边界**递归拆分：<br>  • 偶数个组的块 → 从中间一分为二<br>  • 奇数个组的块 → 遍历所有可能的切分点，选 `max(left_words, right_words)` 最小的那个（左右尽量均衡）<br>  • 只有一个组的块（单个超长组）→ 即使超限也不拆分（不能切断句子内部） |
| **输出** | `blocks`：拆分后的子块列表，每个子块是一个或多个连续长组范围的列表 |
| **函数** | [`_split_block()`](../tools/segment.py) |

```
合并图示：
组: [A_短] [B_长] [C_长] [D_短] [E_长] [F_短]
                         ↓
raw_blocks:    [B+C]              [E]
          （连续长组合并）    （孤立长组单独）

拆分图示（raw block 过大时）：
raw block：[B] [C] [D] [E]  ← 4 个连续长组，总字数超限
                ↓  偶数个组 → 中间一分为二
sub-blocks:  [B+C]        [D+E]
             (左半)        (右半)

raw block：[B] [C] [D] [E] [F]  ← 5 个连续长组，总字数超限
                ↓  奇数个组 → 遍历所有切分点，选左右最均衡的
候选 split: mid=1 → left=1组(B)  right=4组(C+D+E+F)  max=4
            mid=2 → left=2组(B+C) right=3组(D+E+F)    max=3  ← 最优
            mid=3 → left=3组(B+C+D) right=2组(E+F)    max=3  ← 同等最优，取第一个
            mid=4 → left=4组(B+C+D+E) right=1组(F)    max=4
                ↓
sub-blocks:  [B+C]        [D+E+F]
```

**`_find_context()` 对拆分后子块的特殊处理**：未被拆分的块两侧相邻的是短组，直接用作 LLM 上下文。对于拆分后的子块，内部相邻的是同属一个 raw block 的长组兄弟子块——[`_find_context()`](../tools/segment.py) 会将这些相邻长组截断为 `max_context_words` 词（而不是整组传入），提供"接下来是什么"的边界信息，同时避免将整个下一子块的内容塞给 LLM。

### 阶段 4：LLM 标点填充 + 差异分析

| 项目 | 内容 |
|------|------|
| **目的** | 对每个块调用 LLM 添加缺失标点，通过字符差异分析找出新断点 |
| **输入** | `blocks`，`words`，短组（作为只读上下文） |
| **子步骤** | ① 为每个块寻找最近的短组作为上下文（左/右）<br>② 调用 `_llm()` 发送到 LLM，请求添加缺失逗号和句末标点<br>③ `_find_new_breaks()`：对原始文本和 LLM 输出做字符级 diff，识别新增的 .?!<br>④ `_find_new_commas()`：同样 diff 找出新增的逗号<br>⑤ **保护过滤**：拒绝在脆弱尾词（FRAGILE_RE）、短语二元组、加强词 so 前的断点<br>⑥ 将新增逗号注入回单词数据 |
| **输出** | `all_breaks`（原生 + 新增断点的全局单词索引集合） |
| **函数** | [`_llm()`](../tools/segment.py)，[`_find_new_breaks()`](../tools/seg_diff.py)，[`_find_new_commas()`](../tools/seg_diff.py)，[`_find_context()`](../tools/segment.py) |

```
LLM 提示词（默认版）：
  "Fix punctuation in the ASR transcript below. Keep existing punctuation
   that is correct, but ADD missing commas and sentence-ending punctuation
   (. ? !) wherever natural reading requires them..."

差异分析过程：
  原始: "This is great and it works perfectly"
  LLM:  "This is great. And it works perfectly."
                      ↑
                diff 找到新增的 . → 新断点
  保护过滤: 检查断点左侧不以脆弱词结尾、不破坏短语二元组、不是加强词 so
```

### 阶段 5：从所有断点构建段

| 项目 | 内容 |
|------|------|
| **目的** | 在所有累积断点处切分单词列表，构建字幕段 |
| **输入** | `all_breaks`（阶段 4 输出的所有 .?! 断点 + 原生断点），`words` |
| **算法** | 排序所有断点 → 在每个断点处切分 → 构建 `{text, start, end}` 段字典 |
| **输出** | `segments_with_idx`：`(word_start, word_end, segment_dict)` 列表 |
| **函数** | 内联于 [`segment_words()`](../tools/segment.py) |

### 阶段 6：逗号强制切分

| 项目 | 内容 |
|------|------|
| **目的** | 对仍超长的段，在从句内部逗号处切分（而非列表逗号） |
| **输入** | `segments_with_idx`（来自阶段 5），`words`，`min_words`（默认 4） |
| **算法** | 从右到左扫描，找符合条件的逗号：<br>• 逗号两侧各 ≥ `min_words` 个词（逗号词本身不计入任一侧）<br>• 逗号后首词是 `CLAUSE_STARTER`（代词/连词/WH词）或 `ELABORATION_STARTER`（副词/比较/限定词）<br>• 不是列表逗号（`_is_list_comma()` — 右侧有 and/or 且无从句信号）<br>• 找到最右侧符合条件的逗号 → 切分 → 递归处理两侧子段 |
| **输出** | 细化后的段列表 |
| **函数** | [`_comma_split()`](../tools/seg_rules.py)，[`_is_list_comma()`](../tools/seg_rules.py) |

```
示例：
输入: "We have a low pass and a high pass filter control for the noise and the width
       of these are all connected to the oscillator"

在从句逗号处切分:
  ✓ "We have a low pass and a high pass filter control for the noise,"
    └── "and the width of these are all connected to the oscillator"
                                     ↑
                     逗号后 "and" → 从句主语 "the" → 从句边界

跳过列表逗号:
  "parameters, buttons, and knobs" → 不分，因为 "and" 连接列表项
```

**`_CLAUSE_STARTERS` 集合：** 主语代词（I/you/he/she/it/we/they）、指示代词（this/that/these/those/there）、并列连词（and/but/or/so/nor/yet/for）、从属连词（if/when/because/although/since/unless/though/while/where/whereas/as/once/until）、WH-词（which/who/whom/whose/what/how/why/whether）

**`_ELABORATION_STARTERS` 集合：** 比较/相似词（very/similar/much/more）、具体化词（especially/particularly/including/such/like/notably/namely）、范围限定词（mostly/mainly/primarily/largely/typically/generally）、条件/依赖词（depending/based/compared/according/excluding/except/related）、修饰副词（essentially/specifically）

### 阶段 7：连词切分

| 项目 | 内容 |
|------|------|
| **目的** | 对阶段 6 处理后仍超长的段，在引介新分句的并列连词处切分 |
| **输入** | 阶段 6 的输出，`words`，`min_words` |
| **算法** | 两层结构：<br><br>**第一层（规则，无 LLM 调用）：**<br>• `but` → 始终可切分（受 `min_words` 限制）<br>• `so` + `CLAUSE_STARTER` → 切分（连词 so）<br>• `so` + 形容词/副词 → **不**切分（加强词 so，如 "so good"）<br>• `or` + `CLAUSE_STARTER` → 切分<br>• `and` + `CLAUSE_STARTER` → 切分<br><br>**第二层（LLM 辅助）：**<br>• `and`/`so`/`or` + 非 `CLAUSE_STARTER`（且"so"非加强词） → LLM 判断是否连接两个完整分句<br>• 通过 `_classify_conjunctions()` 发送 YES/NO 二分类问题 |
| **输出** | 细化后的段列表 |
| **函数** | [`_conjunction_split()`](../tools/seg_rules.py)，[`_classify_conjunctions()`](../tools/segment.py)，[`_find_ambiguous_conjunctions()`](../tools/seg_rules.py)，[`_is_so_intensifier_target()`](../tools/seg_rules.py) |

```
切分决策树（连词在位置 i 时）：
                      ┌───────────────────────────────────────┐
                      │  连词是 and/so/or/but?                 │
                      └──────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┼────────────────────┐
                    ▼                    ▼                    ▼
                  but                so/or               and
              → 始终切分          ┌─────────────┐   ┌────────────────────┐
                                 │ 后续词是     │   │ 后续词是            │
                                 │ CLAUSE_     │   │ CLAUSE_STARTER?    │
                                 │ STARTER?    │   ├──────┬──────┬──────┤
                                 ├──────┬──────┤   │ 是   │ 否    │ 无法 │
                                 │ 是   │ 否    │  │ 规则  │ LLM  │ 判断 │
                                 ├──────┼──────┤   │ 切分 │ 分类  │ → 不 │
                                 │ 切分 │ 是否  │   │      │ 器   │ 切分 │
                                 │      │ 加强  │   └──────┴──┬───┴──────┘
                                 │      │ so?  │             │
                                 │      ├──┬───┤             ▼
                                 │      │是│ 否│           YES/NO
                                 │      ├──┼───┤
                                 │      │不│LLM │
                                 │      │分│分类│
                                 │      │割│器  │
                                 └──────┴──┴───┘
```

### 阶段 8：LLM 超长句重标点

| 项目 | 内容 |
|------|------|
| **目的** | 对阶段 6+7 都无法切分的超长段，用专门的"拆分长句"提示词重新送 LLM |
| **输入** | 阶段 7 输出中仍超长的段，`words` |
| **提示词** | [`_PHASE8_PROMPT`](../tools/segment.py)：专门要求拆分"and/so/and then"连接的完整意群，在句首连词前加句号 |
| **算法** | ① 用专门的 `_PHASE8_PROMPT` 调用 `_llm()`<br>② `_find_new_breaks()` 找出新增 .?! 断点<br>③ 递归单切分：`_pick_break()` 每次选最平衡（最小化左右词数差）且通过所有保护的断点<br>④ `_split_recursive()` 递归切分两侧，直至所有子段都符合长度限制或无可选断点<br><br>**保护（`_pick_break`）：**<br>• FRAGILE_RE：断点左侧不以脆弱词结尾<br>• 短语二元组：不破坏固定表达<br>• 连词片段：左侧 ≤4 词且以 and/but/so/or 开头 → 拒绝<br>• 加强词 so 检查<br>• 列表枚举保护：断点在 and/or 前且左侧有逗号 → 可能是列表，不切分 |
| **输出** | 细化后的段列表 |
| **函数** | [`_pick_break()`](../tools/segment.py)，[`_split_recursive()`](../tools/segment.py) |

```
与阶段 4 的区别：
  阶段 4：通用标点修复提示词
  阶段 8：专用"拆分超长句"提示词，特别要求拆分 and/so/and then 连缀

递归切分过程：
  段 [ws, we) 仍超长
      │
      ▼
  找出所有候选断点
      │
      ▼
  选最平衡（最小化两侧词数差）+ 通过所有保护
      │
      ├──→ 切分为 [ws, b) + [b, we)
      │                    │
      │            ┌───────┴───────┐
      │            ▼               ▼
      │       递归处理左半      递归处理右半
      │         (≤8层深度)
      │
  若无可选断点 → 保留原段
```

### 阶段 9：LLM 引导的连词片段合并

| 项目 | 内容 |
|------|------|
| **目的** | 将过短的、以连词开头的"寄生片段"合并回前一段 |
| **输入** | 阶段 8 输出的所有段 |
| **条件** | 候选片段：词数 ≤ 8 且首词为 `and/but/so/or`，且前一段不以 .?! 结尾 |
| **算法** | ① 收集所有候选片段 `(idx, seg, first_word)`<br>② 调用 `_classify_conj_merge()`：LLM 判断每个候选是 CONTINUATION（合并） 还是 NEW_SENTENCE（保留）<br>③ 仅当 LLM 确认 CONTINUATION 且合并后不超限时才执行合并 |
| **输出** | 细化后的段列表 |
| **函数** | [`_classify_conj_merge()`](../tools/segment.py) |

```
示例：
  前一段: "So we have a low pass and a high pass filter"
  当前段: "and the width of these are all connected"
                                  ↓
  LLM 判断: CONTINUATION
                                  ↓
  合并后: "So we have a low pass and a high pass filter and
           the width of these are all connected"
  （前提：字数/字符数不超过上限）

硬规则：前一段以 .?! 结尾 → 永不合并（独立句子边界）
```

### 阶段 10：紧急切分

| 项目 | 内容 |
|------|------|
| **目的** | 最后一道防线：对所有前一阶段无法处理的极端超长段进行无 LLM 强制切分 |
| **输入** | 阶段 9 输出中仍超长的段，`words`，`min_words` |
| **条件** | 词数 > `max_words`（30）或字符数 > `max_chars`（120） |
| **算法** | 三轮尝试：<br><br>**第 1 轮：逗号切分**<br>• 扫描所有逗号，选最平衡（最小化左右词数差）且通过列表逗号检查的<br>• 逗号词不计入任一侧的词数<br><br>**第 2 轮：连词/从属连词切分**<br>• `but` → 始终可切分，选最平衡的<br>• `so`/`or` + `CLAUSE_STARTER` → 可切分，选最平衡的<br>• `so`/`or` 非从句主语 → 不在此处切分<br>• `and` → 即使不接 CLAUSE_STARTER 也可切分（比强制中点切分落在脆弱尾词后更好），选最平衡的<br>• 从属连词（because/although/since/unless/while/when/where/if/as）→ 始终可切分，选最平衡的<br><br>**第 3 轮：强制中间切分**<br>• `mid = n // 2`，从中点向两侧扫描，找最安全的切分点（左侧不以 FRAGILE_RE 脆弱词如冠词/介词结尾）<br>• 无安全点则回退到精确中点<br>• 递归处理两侧直至所有段符合限制 |
| **输出** | 最终段列表（切分管道的最终输出） |
| **函数** | [`_phase10_split()`](../tools/segment.py)，[`_phase10_within_limits()`](../tools/segment.py) |


## 关键保护机制

| 保护名称 | 作用阶段 | 用途 |
|----------|----------|------|
| **FRAGILE_RE** | 4, 8 | 拒绝在脆弱尾词（冠词/介词/助动词/情态动词等）后切分 |
| **短语二元组** | 4, 7, 8 | 不破坏 ~5700 条固定表达（如 "such as", "going to", "just so"） |
| **列表逗号检查** | 6, 10 | 不在列表逗号（"a, b and c" 中的逗号）处切分 |
| **加强词 so** | 4, 7, 8 | 不在加强词 so（"so good" 中的程度副词）前切分，只在连词 so（"so I" 中的因果连词）前切分 |
| **连词片段保护** | 8 | 拒绝创建 ≤4 词以连词开头的寄生片段 |
| **列表枚举 and/or** | 7, 8 | 不在列表枚举中的 and/or 前切分 |
| **硬边界（.?!）** | 9 | 前段以句号结尾 → 永不合并跨句边界 |
