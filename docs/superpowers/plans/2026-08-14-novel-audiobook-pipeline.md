# 小说有声书流水线 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把一本小说文本变成按章节合并的多角色有声书：先分析风格与人物，用 IndexTTS-2.5 为每个角色自制音色参考，再按 2.5 约束拆成朗读句并顺序合成。

**Architecture:** 新增独立脚本 `novel_pipeline.py`，复用 `highlight_pipeline` 的 `LLMClient` / JSON 解析 / checkpoint，以及 `dub_pipeline` 的发音标注与情感估计。分析与拆句走 LLM + 确定性规则；音色参考从种子库匹配后由 2.5 生成角色声卡；合成循环调用 `indextts.infer_v2_5.IndexTTS2.infer`；章节用标准 WAV 静音拼接，不走视频/ffmpeg 时间线。

**Tech Stack:** Python 3.10+、IndexTTS-2.5（`indextts.infer_v2_5.IndexTTS2`）、OpenAI-compatible LLM（`LLM_API_KEY`）、PyYAML、soundfile / wave、pytest。不引入 EPUB 解析库，不改 WebUI，不改 `indextts2` CLI。

## Global Constraints

- 引擎固定 **IndexTTS-2.5**：`from indextts.infer_v2_5 import IndexTTS2`，`use_bf16=bool(use_fp16)`，禁止走 `infer_v2`。
- `infer()` 必须传 `lang`（默认 `zh`）；输出 22050 Hz / int16 WAV。
- 参考音频硬截断 **15 秒**；声卡目标 **8–12 秒**；种子至少 **3 秒**。
- 朗读句目标 **20–80 个汉字**，引擎 `max_text_tokens_per_segment=120`；禁止在 `<字|发音>` 标签中间切开。
- 情感向量 8 维，顺序 `[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]`；合成前调用 `tts.normalize_emo_vec`（总和 ≤ 0.8）。
- `duration_factor` 夹在 `0.5–2.0`。旁白默认 `1.05–1.15`，对白按人物设定。
- `use_random=False`。角色声卡生成也不开随机，避免克隆崩坏。
- 命令一律 `uv run` / `PYTHONPATH="$PYTHONPATH:." uv run ...`。
- LLM 只通过 `--llm-api-key` 或 `LLM_API_KEY`；默认 `https://api.openai.com/v1`、`gpt-4o-mini`。
- 中间产物 UTF-8 JSON，`ensure_ascii=False`，`indent=2`。
- 第一版不做 EPUB、不做 m4b/ID3、不加 WebUI 页、不把整本合成一个超大 WAV（只按章输出）。
- 不修改 `dub_pipeline.py` / `highlight_pipeline.py` 的视频逻辑；只 **import** 已有 helper。
- 长篇按章处理：LLM 与 checkpoint 都不把整本塞进单个 JSON 字段。

---

## 方案对比与选定

| 方案 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **A. 独立 `novel_pipeline.py`（选定）** | 与 dub/highlight/intent 同级的第四条流水线 | 与仓库惯例一致；可复用 LLM/发音/TTS 初始化；边界清晰 | 根目录再多一个大脚本 |
| B. 扩 `indextts2 book` | 塞进 `cli_v2.py` | 用户级入口统一 | `cli_v2` 仍绑 IndexTTS-2，没有 `lang`/`duration_factor`；会把分析/LLM 耦进 CLI |
| C. WebUI 新页 | Gradio 向导 | 方便听角色声卡 | 现有 WebUI 是单句演示；长篇 resume/批处理不适合先做 UI |

**音色参考（用户要求「用 2.5 自制」）：**

1. 维护带标签的 **种子音色库**（`examples/voice_*.wav` + `examples/voice_bank.yaml`）。
2. LLM 按人物性别/年龄/气质匹配一枚种子。
3. 用 2.5 合成该角色的 **声卡句**（8–12 秒、音素覆盖好、情绪克制），写入 `voices/{id}.wav`。
4. 后续朗读默认用这张声卡做 `spk_audio_prompt`。
5. `--ref-mode seed` 可改用种子原声（音质更高）；`--voices-dir` 可覆盖某角色 wav。

不采用「无种子纯随机出声」：2.5 是零样本克隆，没有 `spk_audio_prompt` 就无法合成。

---

## File map

| File | Responsibility |
|---|---|
| `novel_pipeline.py` | 整条流水线：解析、分析、配音、拆句、合成、按章合并、CLI |
| `examples/voice_bank.yaml` | 种子音色标签（id / path / gender / age / timbre / languages） |
| `tests/test_novel_pipeline.py` | 章节切分、对话切分、句长、发音保护、情感校验、合并、checkpoint（全部 mock TTS/LLM） |
| `tests/fixtures/novel_sample.txt` | 约 2 章、3 个角色的短样例 |
| `CLAUDE.md` | 增加小说流水线命令与架构说明 |

不新建 `pipelines/` 包。Intent 已经是「独立脚本 + import highlight」模式，这里照做。

复用（只 import，不改行为）：

- `highlight_pipeline`: `LLMClient`, `_parse_json_response`, `_repair_truncated_json`, `_save_checkpoint`, `_load_checkpoint`, `_clear_checkpoint`, `_free_vram`, `_init_tts`, `get_audio_duration`
- `dub_pipeline`: `load_pronunciation_glossary`, `annotate_tts_text`, `estimate_emotion_from_text`
- `indextts.cli_v2`: `_concatenate_wav_segments`（WAV + `silence_after_ms`）

若 `_concatenate_wav_segments` 对跨模块导入过重（它依赖 `cli_v2` 内部 format 字段），则在 `novel_pipeline.py` 内写一个等价的 `_concat_wavs(paths_and_silences, output_path)`，用 `wave` 模块实现，行为与 `cli_v2._write_concat_wav` 相同。

---

## 数据模型

工作目录：`{--work-dir}/{novel_stem}/`，默认 `--work-dir novel_workspace`。

```
novel_workspace/<stem>/
  source.txt
  style.json
  characters.json
  chapters.json
  script/c01.json
  script/c02.json
  voices/narrator.wav
  voices/zhang_san.wav
  voices/manifest.json
  tts/c01/0000.wav
  tts/c02/0000.wav
  chapters/c01.wav
  chapters/c02.wav
  pronunciation.yaml          # optional override
  checkpoint.json
```

`style.json`：

```json
{
  "title": "示例小说",
  "genre": "现代都市",
  "narrative_pov": "第三人称",
  "era": "当代",
  "tone": "克制、略带讽刺",
  "pacing": "中慢",
  "lang": "zh",
  "narrator_style": "沉稳男声，少夸张",
  "duration_factor": 1.08,
  "base_emo": [0, 0, 0, 0, 0, 0.05, 0, 0.35]
}
```

`characters.json` 中每个角色：

```json
{
  "id": "zhang_san",
  "name": "张三",
  "aliases": ["他", "老张"],
  "role": "dialogue",
  "gender": "male",
  "age": "young_adult",
  "personality": "急躁、嘴硬",
  "voice_traits": "偏亮、语速略快",
  "seed_voice_id": "voice_04",
  "ref_wav": "voices/zhang_san.wav",
  "duration_factor": 0.95,
  "base_emo": [0.05, 0.1, 0, 0, 0, 0, 0, 0.2],
  "card_text": "我是张三。这件事我早就看透了，别再跟我绕弯子。"
}
```

旁白固定 `id="narrator"`，`role="narrator"`。

`script/c01.json` 中每个 utterance：

```json
{
  "id": "c01_0007",
  "chapter_id": "c01",
  "seq": 7,
  "speaker_id": "zhang_san",
  "kind": "dialogue",
  "text": "你少来这套。",
  "tts_text": "你少来这套。",
  "lang": "zh",
  "emo_vector": [0.0, 0.2, 0.0, 0.0, 0.05, 0.0, 0.0, 0.15],
  "duration_factor": 0.95,
  "silence_after_ms": 350,
  "wav_path": null
}
```

`kind` ∈ `{narration, dialogue}`。`silence_after_ms` 默认：句内引擎静音 200ms；句间 280；换说话人 420；段末 700；章末不在句上处理。

---

## 流水线步骤

```
小说 TXT
  → Step 0  规范化入库（source.txt）
  → Step 1  切章（规则优先，LLM 只在规则失败时补）
  → Step 2  风格分析（LLM，抽样，不全文）
  → Step 3  人物列表（按章 map-reduce + 合并）
  → Step 4  种子匹配 + 2.5 自制角色声卡
  → Step 5  按章拆成朗读句（规则切引号 + LLM 归属/情感）
  → Step 6  按顺序 2.5 合成（跳过已有 wav）
  → Step 7  按章合并 WAV
```

Checkpoint：`{step, paths}`。每章脚本、每条 wav、每张声卡都以文件是否存在为准，中断后重跑同一命令即可续。

### Step 0 — 入库

- 读 `--input` UTF-8 `.txt` / `.md`。
- 统一换行 `\n`，去掉 UTF-8 BOM。
- Markdown 去掉 `#` 标记但保留标题文字。
- 写入 `source.txt`。

### Step 1 — 切章

规则按顺序匹配：

1. `^第[零一二三四五六七八九十百千0-9]+[章节回卷]`
2. `^Chapter\s+\d+`
3. `^\d+(\.\d+)*\s+\S+` 且行长 < 40
4. 连续 `\n{4,}` 视为弱分隔，仅当全文 > 2 万字且规则 1–3 命中 < 2 章时启用

无匹配则整本 `c01`，标题取文件名。

产出 `chapters.json`：`[{id, index, title, start_char, end_char}]`。`id` 为 `c{index:02d}`。

### Step 2 — 风格

送 LLM：标题候选 + 前 2500 字 + 每章开头 200 字（最多 8 章）+ 结尾 800 字。

Prompt 强制 JSON，字段即 `style.json`。校验：`lang ∈ {zh,en,ja,es,ar}`；`duration_factor` clamp 0.8–1.3；`base_emo` 长度 8、值 ∈ [0,1]。

### Step 3 — 人物

对每章（正文超过 6000 字则再切 4000 字窗口）问 LLM：本章出现的说话人与旁白特征。再合并：

- 同名 / 别名归一
- 旁白只保留一条 `narrator`
- 路人合并为 `crowd`（共用一枚中性种子，不单独出声卡也可）
- 主线角色上限默认 24；超出按出场次数保留，其余并入 `crowd`

### Step 4 — 自制音色参考

`examples/voice_bank.yaml` 描述每条种子。`--voice-bank` 可指向自定义目录（目录内 yaml + wav）。`--narrator-audio` / `--ref-audio` 覆盖旁白种子。

匹配规则（确定性，不用 LLM 再选文件）：

1. `gender` 必须一致（`unknown` 可匹配任意）
2. `age` 相同优先
3. `timbre` 与 `voice_traits` 关键词打分
4. 已被占用的种子降权，避免全员同声
5. 并列时按 yaml 顺序

声卡文本：优先用角色 `card_text`；否则模板：

```
我是{name}。{one_sentence_personality}。今天天气不错，山上的风从松树林里穿过来，溪水轻轻响着。一二三四五，金木水火土。
```

合成：

```python
tts.infer(
    spk_audio_prompt=seed_path,
    text=card_text,
    lang=style["lang"],
    output_path=ref_wav,
    emo_vector=tts.normalize_emo_vec(character["base_emo"]),
    duration_factor=1.0,
    interval_silence=200,
    max_text_tokens_per_segment=120,
    use_random=False,
    verbose=False,
)
```

声卡 wav 已存在则跳过。`--ref-mode seed` 时把种子复制（或直接记录路径）为 `ref_wav`，仍写出 `voices/manifest.json`。

### Step 5 — 拆朗读句（按 2.5 特点）

**5a 规则切分（不丢字）**

- 保护已有 `<...|...>`。
- 中文引号 `「」` `『』` `“”` `‘’` `"..."` 抽出为 `dialogue`，前后叙述为 `narration`。
- 引导语「张三说：」留在 narration；引号内归 dialogue。
- 再按 `。！？；…` 断句；逗号只在当前块 > 80 字时断。
- 硬限制 80 字；超限且无标点则按 80 切开，但跳过标签内部。

**5b LLM 归属（按章）**

输入：本章 utterance 的 `text` + `kind` + 人物表。输出：`speaker_id`、`emo_vector`、可选改写后的 `tts_text`。

`tts_text` 约束（写进 prompt，并在代码再校验）：

- 口语化，保留原意，不扩写剧情
- 不发明旁白评论
- 数字/英文专有名词可加 `<ChatGPT|...>` 式标注
- 单条不超过 80 字
- 不删除原文信息点

失败回退：dialogue 无主则 `narrator`；`emo_vector` 用 `estimate_emotion_from_text(text)` 或角色 `base_emo`；`tts_text = text`。

**5c 2.5 适配层 `prepare_tts_text(text, work_dir)`**

1. `annotate_tts_text(text, glossary=load_pronunciation_glossary(work_dir=work_dir))`
2. 确认标签完整
3. 夹 `duration_factor`、补 `lang`、算 `silence_after_ms`

说话人切换：`silence_after_ms = 420`；同说话人：`280`；`kind` 从 dialogue 回到 narration：`350`。

### Step 6 — 顺序合成

只在此时加载 TTS（若 Step 4 已加载则复用，章间不卸载）。每条：

```python
tts.infer(
    spk_audio_prompt=characters[utt["speaker_id"]]["ref_wav"],
    text=utt["tts_text"],
    output_path=f"{work_dir}/tts/{chapter_id}/{utt['seq']:04d}.wav",
    lang=utt["lang"],
    emo_vector=tts.normalize_emo_vec(utt["emo_vector"]) if any(utt["emo_vector"]) else None,
    duration_factor=clamp(utt["duration_factor"], 0.5, 2.0),
    interval_silence=200,
    max_text_tokens_per_segment=120,
    use_random=False,
    verbose=False,
)
```

已有 wav 则跳过并回填时长。失败：写等长静音（`max(1.2, 0.15 * len(tts_text))` 秒），打印 warning，不中断整章。`--strict` 时失败即退出。

每 20 条刷新 `script/{chapter_id}.json` 的 `wav_path`。

### Step 7 — 按章合并

```python
_concat_wavs(
    [{"audio_path": u["wav_path"], "silence_after_ms": u["silence_after_ms"]}
     for u in chapter_utterances],
    output_path=f"{work_dir}/chapters/{chapter_id}.wav",
)
```

最终把各章 wav 复制（或列出）到 `--output`：

- `--output` 是目录：`{output}/{stem}_c01.wav`, `...`
- `--output` 是文件且只有一章：写该文件
- `--output` 是文件且多章：写成目录 `{stem}_chapters/`，并打印清单；**不**默认拼全书

可选 `--concat-book` 再拼 `book.wav`（章间静音 1500ms）。第一版实现该开关，默认关。

---

## CLI

```bash
PYTHONPATH="$PYTHONPATH:." uv run novel_pipeline.py novel.txt \
  --ref-audio examples/voice_01.wav \
  -o audiobook_out \
  --llm-api-key "$LLM_API_KEY"

# 只分析到人物表
PYTHONPATH="$PYTHONPATH:." uv run novel_pipeline.py novel.txt --stop-after characters

# 用种子原声，不自制声卡
PYTHONPATH="$PYTHONPATH:." uv run novel_pipeline.py novel.txt --ref-mode seed --ref-audio examples/voice_01.wav

# 重合成第 3 章
PYTHONPATH="$PYTHONPATH:." uv run novel_pipeline.py novel.txt --chapter 3 --force-tts

# 中断后续跑同一命令即可
```

参数：`--input` 位置参数、`--output/-o`、`--work-dir`、`--model-dir`（默认 `checkpoints`）、`--fp16`（实际 `use_bf16`）、`--ref-audio`、`--narrator-audio`、`--voice-bank`、`--ref-mode {card,seed}`、`--lang`、`--llm-api-key`、`--llm-api-base`、`--llm-model`、`--stop-after {chapters,style,characters,voices,script,tts,merge}`、`--chapter`、`--force-tts`、`--strict`、`--concat-book`、`--cleanup`。

---

### Task 1: 入库、切章、静音拼接

**Files:**
- Create: `novel_pipeline.py`
- Create: `tests/fixtures/novel_sample.txt`
- Create: `tests/test_novel_pipeline.py`

**Interfaces:**
- Consumes: UTF-8 小说文本
- Produces:
  - `ingest_text(raw: str) -> str`
  - `split_chapters(text: str, source_name: str) -> list[dict]`
  - `_concat_wavs(segments: list[dict], output_path: str) -> str`
  - `CHAPTER_RE` 与 `id` 格式 `c01`

- [ ] **Step 1: 写失败测试**

`tests/fixtures/novel_sample.txt`：

```text
第一章 雨夜
张三站在巷口。他说：「今晚别跟过来。」
李四笑了笑：「我偏要来。」

第二章 黎明
雨停了。张三把刀收回袖中。
```

`tests/test_novel_pipeline.py` 先测：

```python
from novel_pipeline import ingest_text, split_chapters


def test_split_chapters_chinese_heading():
    text = open("tests/fixtures/novel_sample.txt", encoding="utf-8").read()
    chapters = split_chapters(ingest_text(text), "novel_sample.txt")
    assert [c["id"] for c in chapters] == ["c01", "c02"]
    assert chapters[0]["title"].startswith("第一章")
    assert "张三站在巷口" in text[chapters[0]["start_char"]:chapters[0]["end_char"]]
    assert "雨停了" in text[chapters[1]["start_char"]:chapters[1]["end_char"]]


def test_ingest_strips_bom_and_crlf():
    assert ingest_text("\ufeffhello\r\nworld\r\n") == "hello\nworld\n"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_split_chapters_chinese_heading -v`

Expected: FAIL，`novel_pipeline` 未定义。

- [ ] **Step 3: 实现 `ingest_text` / `split_chapters` / `_concat_wavs`**

`split_chapters` 用 `re.finditer(r'(?m)^(第[零一二三四五六七八九十百千万0-9]+[章节回卷][^\n]*)', text)`。无匹配则返回单章 `c01`。

`_concat_wavs` 用标准库 `wave`：逐段写入 PCM，并在段后追加 `silence_after_ms` 的零字节。假设输入都是 22050/mono/16-bit；不一致则 resample 或报错（测试里写同格式）。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_split_chapters_chinese_heading tests/test_novel_pipeline.py::test_ingest_strips_bom_and_crlf -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add novel_pipeline.py tests/test_novel_pipeline.py tests/fixtures/novel_sample.txt
git commit -m "feat: add novel chapter ingest and split"
```

---

### Task 2: 对话/旁白切分与 2.5 句长约束

**Files:**
- Modify: `novel_pipeline.py`
- Modify: `tests/test_novel_pipeline.py`

**Interfaces:**
- Consumes: 一章纯文本
- Produces:
  - `split_utterances(chapter_text: str, chapter_id: str) -> list[dict]`
  - `enforce_tts_limits(text: str, max_chars: int = 80) -> list[str]`
  - utterance 必有 `id, chapter_id, seq, kind, text, tts_text, lang`

- [ ] **Step 1: 写失败测试**

```python
from novel_pipeline import split_utterances, enforce_tts_limits


def test_split_utterances_quotes():
    utts = split_utterances("张三说：「今晚别跟过来。」巷子里很静。", "c01")
    kinds = [u["kind"] for u in utts]
    assert "dialogue" in kinds and "narration" in kinds
    dialogue = next(u for u in utts if u["kind"] == "dialogue")
    assert "今晚别跟过来" in dialogue["text"]
    assert "「" not in dialogue["tts_text"] and "」" not in dialogue["tts_text"]


def test_enforce_tts_limits_protects_pron_tags():
    text = "他在银<行|HANG2>办了一件非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常非常长的业务。"
    parts = enforce_tts_limits(text, max_chars=20)
    assert all("<行|HANG2>" in p or "<行|HANG2>" not in text for p in parts) or any("<行|HANG2>" in p for p in parts)
    assert all("<行|" not in p or "|HANG2>" in p for p in parts)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_split_utterances_quotes tests/test_novel_pipeline.py::test_enforce_tts_limits_protects_pron_tags -v`

Expected: FAIL

- [ ] **Step 3: 实现切分**

引号配对用栈，支持 `「」『』“”‘’"`。对白 `tts_text` 去掉包裹引号。`enforce_tts_limits` 先把 `<[^|>]+\\|[^>]+>` 换成占位符再按标点/字数切，最后还原。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_split_utterances_quotes tests/test_novel_pipeline.py::test_enforce_tts_limits_protects_pron_tags -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add novel_pipeline.py tests/test_novel_pipeline.py
git commit -m "feat: split novel narration and dialogue for IndexTTS 2.5"
```

---

### Task 3: 风格 / 人物 LLM 分析与校验

**Files:**
- Modify: `novel_pipeline.py`
- Modify: `tests/test_novel_pipeline.py`

**Interfaces:**
- Consumes: `LLMClient.chat(prompt) -> str`；`chapters` + `source`
- Produces:
  - `analyze_style(text, chapters, llm) -> dict`
  - `extract_characters(text, chapters, llm) -> list[dict]`
  - `merge_character_lists(batches: list[list[dict]]) -> list[dict]`
  - `validate_style(data: dict) -> dict`
  - `validate_character(data: dict) -> dict`

- [ ] **Step 1: 写失败测试**

```python
from novel_pipeline import merge_character_lists, validate_style, validate_character


def test_merge_characters_aliases_and_narrator():
    merged = merge_character_lists([
        [{"id": "zhang_san", "name": "张三", "aliases": ["老张"], "role": "dialogue",
          "gender": "male", "age": "young_adult", "personality": "急", "voice_traits": "亮"}],
        [{"id": "lao_zhang", "name": "老张", "aliases": ["张三"], "role": "dialogue",
          "gender": "male", "age": "young_adult", "personality": "急躁", "voice_traits": "偏亮"}],
        [{"id": "narrator", "name": "旁白", "aliases": [], "role": "narrator",
          "gender": "male", "age": "middle", "personality": "沉稳", "voice_traits": "低"}],
    ])
    names = {c["name"] for c in merged}
    assert "张三" in names or "老张" in names
    assert sum(1 for c in merged if c["role"] == "narrator") == 1
    assert len([c for c in merged if c["name"] in {"张三", "老张"}]) == 1


def test_validate_style_clamps_duration_and_emo():
    style = validate_style({
        "title": "x", "genre": "y", "narrative_pov": "第三人称", "era": "当代",
        "tone": "冷", "pacing": "慢", "lang": "ZH", "narrator_style": "沉",
        "duration_factor": 3.0, "base_emo": [1, 1, 1],
    })
    assert style["lang"] == "zh"
    assert 0.8 <= style["duration_factor"] <= 1.3
    assert len(style["base_emo"]) == 8
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_merge_characters_aliases_and_narrator tests/test_novel_pipeline.py::test_validate_style_clamps_duration_and_emo -v`

Expected: FAIL

- [ ] **Step 3: 实现分析函数**

`analyze_style` / `extract_characters` 使用从 `highlight_pipeline` 导入的 `LLMClient` 与 `_parse_json_response`。Prompt 要求「只输出 JSON，不要 markdown」。`merge_character_lists` 用名称+别名的无向连通分量合并。保证结果里始终有 `narrator`。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_merge_characters_aliases_and_narrator tests/test_novel_pipeline.py::test_validate_style_clamps_duration_and_emo -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add novel_pipeline.py tests/test_novel_pipeline.py
git commit -m "feat: analyze novel style and merge character lists"
```

---

### Task 4: 种子库匹配与 2.5 角色声卡

**Files:**
- Create: `examples/voice_bank.yaml`
- Modify: `novel_pipeline.py`
- Modify: `tests/test_novel_pipeline.py`

**Interfaces:**
- Consumes: `characters`, `voice_bank.yaml`, `IndexTTS2.infer`
- Produces:
  - `load_voice_bank(path: str) -> list[dict]`
  - `assign_seed_voices(characters, bank, narrator_audio=None) -> list[dict]`
  - `build_card_text(character: dict) -> str`
  - `generate_character_voices(characters, tts, work_dir, lang, ref_mode) -> list[dict]`

- [ ] **Step 1: 写失败测试与种子表**

`examples/voice_bank.yaml`：

```yaml
voices:
  - id: voice_01
    path: examples/voice_01.wav
    gender: female
    age: young_adult
    timbre: bright
    languages: [ZH, EN]
  - id: voice_04
    path: examples/voice_04.wav
    gender: male
    age: young_adult
    timbre: firm
    languages: [ZH]
  - id: voice_05
    path: examples/voice_05.wav
    gender: male
    age: middle
    timbre: deep
    languages: [ZH]
```

测试只测匹配与声卡文本，TTS 用假对象：

```python
from novel_pipeline import assign_seed_voices, build_card_text, generate_character_voices


def test_assign_seed_voices_prefers_gender_and_avoids_collision():
    bank = [
        {"id": "voice_04", "path": "a.wav", "gender": "male", "age": "young_adult", "timbre": "firm"},
        {"id": "voice_05", "path": "b.wav", "gender": "male", "age": "middle", "timbre": "deep"},
        {"id": "voice_01", "path": "c.wav", "gender": "female", "age": "young_adult", "timbre": "bright"},
    ]
    chars = [
        {"id": "narrator", "name": "旁白", "role": "narrator", "gender": "male", "age": "middle", "voice_traits": "低沉"},
        {"id": "zhang_san", "name": "张三", "role": "dialogue", "gender": "male", "age": "young_adult", "voice_traits": "硬"},
        {"id": "li_si", "name": "李四", "role": "dialogue", "gender": "female", "age": "young_adult", "voice_traits": "亮"},
    ]
    out = assign_seed_voices(chars, bank)
    seeds = {c["id"]: c["seed_voice_id"] for c in out}
    assert seeds["narrator"] == "voice_05"
    assert seeds["zhang_san"] == "voice_04"
    assert seeds["li_si"] == "voice_01"
    assert len(set(seeds.values())) == 3


def test_generate_character_voices_writes_card(tmp_path):
    recorded = []
    class FakeTTS:
        def normalize_emo_vec(self, v):
            return v
        def infer(self, **kwargs):
            recorded.append(kwargs)
            open(kwargs["output_path"], "wb").write(b"RIFF")
    chars = [{
        "id": "zhang_san", "name": "张三", "personality": "急躁",
        "seed_path": "seed.wav", "base_emo": [0, 0.1, 0, 0, 0, 0, 0, 0.2],
        "duration_factor": 0.95, "card_text": None,
    }]
    out = generate_character_voices(chars, FakeTTS(), str(tmp_path), "zh", "card")
    assert recorded[0]["lang"] == "zh"
    assert recorded[0]["spk_audio_prompt"] == "seed.wav"
    assert out[0]["ref_wav"].endswith("zhang_san.wav")
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_assign_seed_voices_prefers_gender_and_avoids_collision tests/test_novel_pipeline.py::test_generate_character_voices_writes_card -v`

Expected: FAIL

- [ ] **Step 3: 实现匹配与声卡生成**

`assign_seed_voices` 先分配 narrator。`generate_character_voices` 在 `card` 模式调用 `tts.infer(..., lang=lang, use_random=False)`；`seed` 模式 `shutil.copy2(seed_path, ref_wav)`。写出 `voices/manifest.json`。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_assign_seed_voices_prefers_gender_and_avoids_collision tests/test_novel_pipeline.py::test_generate_character_voices_writes_card -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add novel_pipeline.py examples/voice_bank.yaml tests/test_novel_pipeline.py
git commit -m "feat: assign seeds and synthesize per-character voice cards"
```

---

### Task 5: 朗读脚本归属、情感与发音

**Files:**
- Modify: `novel_pipeline.py`
- Modify: `tests/test_novel_pipeline.py`

**Interfaces:**
- Consumes: `split_utterances` 结果 + `characters` + `style` + LLM
- Produces:
  - `build_chapter_script(chapter_text, chapter_id, characters, style, llm, work_dir) -> list[dict]`
  - `assign_silence(utterances) -> list[dict]`
  - `prepare_tts_text(text, work_dir) -> str`

- [ ] **Step 1: 写失败测试**

```python
from novel_pipeline import assign_silence, prepare_tts_text


def test_assign_silence_longer_on_speaker_change():
    utts = [
        {"speaker_id": "narrator", "kind": "narration"},
        {"speaker_id": "zhang_san", "kind": "dialogue"},
        {"speaker_id": "zhang_san", "kind": "dialogue"},
        {"speaker_id": "narrator", "kind": "narration"},
    ]
    out = assign_silence(utts)
    assert out[0]["silence_after_ms"] == 420
    assert out[1]["silence_after_ms"] == 280
    assert out[2]["silence_after_ms"] == 350


def test_prepare_tts_text_applies_glossary(tmp_path, monkeypatch):
    (tmp_path / "pronunciation.yaml").write_text("银行: 银<行|HANG2>\n", encoding="utf-8")
    text = prepare_tts_text("他去银行了", str(tmp_path))
    assert "<行|HANG2>" in text
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_assign_silence_longer_on_speaker_change tests/test_novel_pipeline.py::test_prepare_tts_text_applies_glossary -v`

Expected: FAIL

- [ ] **Step 3: 实现脚本层**

`build_chapter_script`：规则切分 → 按每 40 条一批问 LLM 填 `speaker_id`/`emo_vector`/`tts_text` → 校验 speaker 必须在人物表，否则 `narrator` → `prepare_tts_text` → 用角色 `duration_factor` 覆盖 → `assign_silence`。LLM 失败用 `estimate_emotion_from_text`。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_assign_silence_longer_on_speaker_change tests/test_novel_pipeline.py::test_prepare_tts_text_applies_glossary -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add novel_pipeline.py tests/test_novel_pipeline.py
git commit -m "feat: build per-chapter reading script with emotion and glossary"
```

---

### Task 6: 顺序合成与按章合并

**Files:**
- Modify: `novel_pipeline.py`
- Modify: `tests/test_novel_pipeline.py`

**Interfaces:**
- Consumes: chapter scripts + character `ref_wav` + `IndexTTS2`
- Produces:
  - `synthesize_chapter(utterances, characters, tts, work_dir, strict=False) -> list[dict]`
  - `merge_chapter(utterances, output_path) -> str`
  - `synthesize` 必须传 `lang`，必须 `normalize_emo_vec`

- [ ] **Step 1: 写失败测试**

```python
from novel_pipeline import synthesize_chapter, merge_chapter
import wave, struct


def _write_wav(path, nframes=2205):
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(22050)
        w.writeframes(b"\x00\x00" * nframes)


def test_synthesize_chapter_passes_v25_kwargs(tmp_path):
    recorded = []
    class FakeTTS:
        def normalize_emo_vec(self, v):
            return [min(x, 0.8) for x in v]
        def infer(self, **kwargs):
            recorded.append(kwargs)
            _write_wav(kwargs["output_path"])
    chars = {"zhang_san": {"ref_wav": "ref.wav", "duration_factor": 0.95}}
    utts = [{
        "id": "c01_0000", "chapter_id": "c01", "seq": 0,
        "speaker_id": "zhang_san", "kind": "dialogue",
        "text": "别过来。", "tts_text": "别过来。", "lang": "zh",
        "emo_vector": [0, 0.2, 0, 0, 0, 0, 0, 0.1],
        "duration_factor": 0.95, "silence_after_ms": 280,
    }]
    out = synthesize_chapter(utts, chars, FakeTTS(), str(tmp_path))
    assert recorded[0]["lang"] == "zh"
    assert recorded[0]["spk_audio_prompt"] == "ref.wav"
    assert recorded[0]["duration_factor"] == 0.95
    assert out[0]["wav_path"].endswith("0000.wav")


def test_synthesize_skips_existing(tmp_path):
    tts_dir = tmp_path / "tts" / "c01"
    tts_dir.mkdir(parents=True)
    _write_wav(tts_dir / "0000.wav")
    class BoomTTS:
        def infer(self, **kwargs):
            raise AssertionError("should skip")
        def normalize_emo_vec(self, v):
            return v
    utts = [{
        "id": "c01_0000", "chapter_id": "c01", "seq": 0,
        "speaker_id": "zhang_san", "tts_text": "x", "lang": "zh",
        "emo_vector": [0]*8, "duration_factor": 1.0, "silence_after_ms": 200,
    }]
    synthesize_chapter(utts, {"zhang_san": {"ref_wav": "ref.wav"}}, BoomTTS(), str(tmp_path))


def test_merge_chapter_inserts_silence(tmp_path):
    a = tmp_path / "a.wav"; b = tmp_path / "b.wav"; out = tmp_path / "c.wav"
    _write_wav(a, 2205); _write_wav(b, 2205)
    merge_chapter(
        [{"wav_path": str(a), "silence_after_ms": 1000},
         {"wav_path": str(b), "silence_after_ms": 0}],
        str(out),
    )
    with wave.open(str(out)) as w:
        assert w.getnframes() == 2205 + 22050 + 2205
```

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_synthesize_chapter_passes_v25_kwargs tests/test_novel_pipeline.py::test_synthesize_skips_existing tests/test_novel_pipeline.py::test_merge_chapter_inserts_silence -v`

Expected: FAIL

- [ ] **Step 3: 实现合成与合并**

`synthesize_chapter` 写到 `{work_dir}/tts/{chapter_id}/{seq:04d}.wav`。`merge_chapter` 调 `_concat_wavs`。TTS 初始化复用 `_init_tts(model_dir, use_fp16)`。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py -k "synthesize or merge_chapter" -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add novel_pipeline.py tests/test_novel_pipeline.py
git commit -m "feat: synthesize novel lines and merge chapter wavs"
```

---

### Task 7: 编排、checkpoint、CLI

**Files:**
- Modify: `novel_pipeline.py`
- Modify: `tests/test_novel_pipeline.py`
- Modify: `CLAUDE.md`

**Interfaces:**
- Consumes: 以上全部函数
- Produces:
  - `run_novel_pipeline(input_path, output, work_dir, ...) -> dict`
  - `main()` argparse
  - checkpoint `novel_workspace/<stem>/checkpoint.json`，字段 `{step, paths}`
  - `step` ∈ `0 ingest … 7 merge`

- [ ] **Step 1: 写失败测试**

```python
from novel_pipeline import run_novel_pipeline


def test_run_pipeline_stop_after_chapters(tmp_path):
    src = tmp_path / "book.txt"
    src.write_text("第一章 一\n你好。\n第二章 二\n再见。\n", encoding="utf-8")
    result = run_novel_pipeline(
        input_path=str(src),
        output=str(tmp_path / "out"),
        work_dir=str(tmp_path / "ws"),
        stop_after="chapters",
        llm_api_key="dummy",
    )
    assert result["step"] == 1
    assert (tmp_path / "ws" / "book" / "chapters.json").is_file()
```

再加一条 CLI smoke：`test_cli_requires_llm_for_full_run` 用 `argparse` 解析 `--help` 含 `--ref-mode`。

- [ ] **Step 2: 跑测试确认失败**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py::test_run_pipeline_stop_after_chapters -v`

Expected: FAIL

- [ ] **Step 3: 实现 `run_novel_pipeline` 与 `main`**

步骤闸门：

```python
if done < 1: chapters = split...; save; done = 1
if stop_after == "chapters": return
...
```

LLM key：`--stop-after` 属于 `{chapters}` 时不强制；`style` 及之后强制。TTS 在 voices/tts 步加载，结束 `_free_vram`。`--chapter N` 只处理该章的 script/tts/merge。`--force-tts` 删除目标章 `tts/` 后重合成。`--cleanup` 成功后删 work dir，但保留 `--output` 中的章 wav。

`CLAUDE.md` 增加：

```bash
PYTHONPATH="$PYTHONPATH:." uv run novel_pipeline.py novel.txt --ref-audio voice.wav -o audiobook_out
```

以及架构段：小说 TXT → 风格/人物 → 2.5 声卡 → 朗读句 → 按章 WAV。

- [ ] **Step 4: 跑测试确认通过**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_novel_pipeline.py -v`

Expected: 全 PASS

- [ ] **Step 5: Commit**

```bash
git add novel_pipeline.py tests/test_novel_pipeline.py CLAUDE.md
git commit -m "feat: orchestrate novel-to-audiobook pipeline with resume"
```

---

### Task 8: 回归与 2.5 接线检查

**Files:**
- Modify: `tests/test_pipeline_tts_v25.py`
- Modify: `tests/test_novel_pipeline.py`（如需）

**Interfaces:**
- Consumes: `novel_pipeline._init_tts`（直接用 highlight 的 `_init_tts`，或本模块包装）
- Produces: 与 dub/highlight 相同的「2.5 + bf16 + lang」断言

- [ ] **Step 1: 扩展 `tests/test_pipeline_tts_v25.py`**

```python
from novel_pipeline import synthesize_chapter


def test_novel_synthesize_uses_lang_zh(tmp_path, monkeypatch):
    recorded = {}
    class FakeTTS:
        def normalize_emo_vec(self, vec):
            return vec
        def infer(self, **kwargs):
            recorded.update(kwargs)
            p = kwargs["output_path"]
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_bytes(b"RIFF")
    utts = [{
        "chapter_id": "c01", "seq": 0, "speaker_id": "narrator",
        "tts_text": "你好", "lang": "zh", "emo_vector": [0]*8,
        "duration_factor": 1.0, "silence_after_ms": 200,
    }]
    synthesize_chapter(utts, {"narrator": {"ref_wav": "ref.wav"}}, FakeTTS(), str(tmp_path))
    assert recorded.get("lang") == "zh"
```

- [ ] **Step 2: 跑测试确认（实现后应 PASS）**

Run: `PYTHONPATH="$PYTHONPATH:." uv run pytest tests/test_pipeline_tts_v25.py tests/test_novel_pipeline.py -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_pipeline_tts_v25.py tests/test_novel_pipeline.py
git commit -m "test: assert novel pipeline talks to IndexTTS 2.5"
```

---

## 明确不做（第一版）

- EPUB / 网页抓取
- 全书一个 m4b / 章节元数据播放列表（可用 `--concat-book` 得到单 WAV）
- WebUI 角色试听页
- 背景乐、环境音、多麦混响
- 自动从零生成「全新人声」而不提供任何种子 wav
- 改 `indextts2` CLI 去支持 2.5 的 `lang`

---

## Spec coverage

| 需求 | 任务 |
|---|---|
| 分析小说整体风格 | Task 3 `analyze_style` |
| 每个人物特点 + 人物列表 | Task 3 `extract_characters` / `merge_character_lists` |
| 用 2.5 为角色自制音色参考 | Task 4 `generate_character_voices` |
| 按 2.5 特点拆朗读内容 | Task 2 + Task 5 |
| 按顺序生成音频 | Task 6 `synthesize_chapter` |
| 按章节合并 | Task 6 `merge_chapter` + Task 7 输出 |

## Placeholder scan

无 TBD / 无「稍后实现」。声卡在没有种子时的行为已写死：必须提供 `--ref-audio` 或可用的 `voice_bank.yaml`。

## Type consistency

- 角色主键始终 `id`（旁白 `narrator`）
- 章主键始终 `c01` 格式
- utterance 主键 `{chapter_id}_{seq:04d}`
- 合成参数名与 `IndexTTS2.infer` 一致：`spk_audio_prompt`, `lang`, `emo_vector`, `duration_factor`
