"""
Intent-Driven Video Content Pipeline

An intent-driven video content creation engine. The user provides a video +
natural language intent (e.g., "分析演讲技巧") and it creates a short video
with narration, fancy text effects, and emotional TTS.

Pipeline:
    Step 0: Intent planning (LLM analyzes intent → creates plan)
    Step 1: Extract audio (ffmpeg)
    Step 2: Speech transcription (WhisperX)
    Step 3: Video analysis with intent focus (Qwen2.5-VL)
    Step 4: Intent-driven script generation + fancy text choreography (LLM API)
    Step 5: Narration synthesis with emotion profile (IndexTTS2)
    Step 6: Video assembly + fancy text rendering (ffmpeg + ASS)

Usage:
    # Single video
    python intent_pipeline.py video.mp4 --intent "分析演讲技巧" --ref-audio voice.wav -o output.mp4

    # Batch mode
    python intent_pipeline.py --batch /path/to/videos --intent "..." --ref-audio voice.wav -o /output

    # Resume after interruption (auto-detected)
    python intent_pipeline.py video.mp4 --intent "分析演讲技巧" --ref-audio voice.wav

Requires: whisperx, transformers, qwen-vl-utils, openai, ffmpeg
"""

import argparse
import dataclasses
import gc
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from highlight_pipeline import (
    _run_ffmpeg, _run_ffprobe, _get_video_info, get_audio_duration,
    _free_vram, _detect_device, _escape_ffmpeg_text,
    _parse_json_response, _repair_truncated_json,
    _format_ass_time, _get_color_grade_filter,
    _compute_timeline, _build_narration_track,
    _save_checkpoint, _load_checkpoint, _clear_checkpoint,
    extract_audio, transcribe_audio, generate_title_cards, concat_with_transitions,
    LLMClient, TRANSITIONS, COLOR_GRADES, EMOTION_DIMS,
    X264_ARGS, AAC_ARGS, TTS_SAMPLE_RATE, CROSSFADE_DURATION,
    NARRATION_VOLUME, ORIGINAL_VOLUME, TITLE_CARD_DURATION,
    VL_FPS, VL_TOTAL_PIXELS, VL_MIN_PIXELS, VL_MAX_PIXELS, VL_MAX_FRAMES,
    MIN_IMPORTANCE, _init_tts,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTENT_CHECKPOINT_FILE = "intent_checkpoint.json"

ORIENTATIONS = {
    "portrait": (1080, 1920),
    "landscape": (1920, 1080),
}

FANCY_TEXT_EFFECTS = [
    "pop_zoom", "slide_in_left", "slide_in_right", "slide_in_top", "slide_in_bottom",
    "fade", "word_highlight", "typewriter", "scroll_marquee", "outlined",
    "gradient_color", "shadow_3d", "rotate_in", "bounce", "multi_layer",
    "emphasis_glow", "shake", "subtitle_default",
]

_NAMED_COLORS = {
    "white": "&HFFFFFF&",
    "cyan": "&HFFFF00&",
    "red": "&H0000FF&",
    "yellow": "&H00FFFF&",
    "green": "&H00FF00&",
    "blue": "&HFF0000&",
    "orange": "&H0080FF&",
    "pink": "&HCB69FF&",
}


# ---------------------------------------------------------------------------
# IntentPlan dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class IntentPlan:
    intent_summary: str
    content_style: str  # educational|emotional|analytical|entertaining|motivational
    analysis_dimensions: list
    narrative_structure: str  # progressive|comparative|listicle|story_arc
    narrative_description: str
    persona: str
    persona_prompt: str
    tone_keywords: list
    hook_strategy: str
    emotion_profile: dict  # {base_vector: [8 floats], allow_high_emotion: bool}
    fancy_text_guidance: dict  # {density: str, style_tone: str}
    title_suggestion: str


# ---------------------------------------------------------------------------
# Step 0: Intent Planning
# ---------------------------------------------------------------------------

def plan_intent(intent, llm_client):
    """Analyze user intent and create a detailed content plan via LLM."""
    prompt = f"""你是一位专业的内容策划师。用户想要基于一个视频创作短视频内容。

用户意图：{intent}

请分析用户的创作意图，制定详细的创作计划。返回严格的JSON（不要markdown围栏）：
{{
  "intent_summary": "一句话概括用户想做什么",
  "content_style": "educational 或 emotional 或 analytical 或 entertaining 或 motivational",
  "analysis_dimensions": ["需要在视频中重点关注的维度，3-6个"],
  "narrative_structure": "progressive 或 comparative 或 listicle 或 story_arc",
  "narrative_description": "叙事结构的具体说明（2-3句）",
  "persona": "生成脚本时的角色名称，如'资深演讲教练'",
  "persona_prompt": "完整的角色设定描述（2-3句，用于指导后续脚本生成）",
  "tone_keywords": ["3-5个语气关键词，如'专业','简洁','有洞察力'"],
  "hook_strategy": "开头吸引观众的策略（1-2句）",
  "emotion_profile": {{
    "base_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "allow_high_emotion": false
  }},
  "fancy_text_guidance": {{
    "density": "low 或 medium 或 high",
    "style_tone": "clean 或 playful 或 dramatic 或 professional"
  }},
  "title_suggestion": "推荐的视频标题"
}}"""

    print("  Planning intent...")
    response = llm_client.chat(prompt, temperature=0.3)
    data = _parse_json_response(response)
    if data is None:
        raise RuntimeError(f"Failed to parse intent plan response:\n{response[:500]}")

    # Validate content_style
    valid_styles = {"educational", "emotional", "analytical", "entertaining", "motivational"}
    if data.get("content_style") not in valid_styles:
        data["content_style"] = "educational"

    # Validate narrative_structure
    valid_structures = {"progressive", "comparative", "listicle", "story_arc"}
    if data.get("narrative_structure") not in valid_structures:
        data["narrative_structure"] = "progressive"

    # Validate emotion_profile
    ep = data.get("emotion_profile", {})
    base_vec = ep.get("base_vector", [0.0] * 8)
    if not isinstance(base_vec, list) or len(base_vec) != 8:
        base_vec = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
    base_vec = [max(0.0, min(1.0, float(v))) for v in base_vec[:8]]
    while len(base_vec) < 8:
        base_vec.append(0.0)
    data["emotion_profile"] = {
        "base_vector": base_vec,
        "allow_high_emotion": bool(ep.get("allow_high_emotion", False)),
    }

    # Validate fancy_text_guidance
    ftg = data.get("fancy_text_guidance", {})
    valid_densities = {"low", "medium", "high"}
    if ftg.get("density") not in valid_densities:
        ftg["density"] = "medium"
    valid_style_tones = {"clean", "playful", "dramatic", "professional"}
    if ftg.get("style_tone") not in valid_style_tones:
        ftg["style_tone"] = "clean"
    data["fancy_text_guidance"] = ftg

    # Ensure list fields
    if not isinstance(data.get("analysis_dimensions"), list):
        data["analysis_dimensions"] = ["内容质量", "情感表达", "视觉呈现"]
    if not isinstance(data.get("tone_keywords"), list):
        data["tone_keywords"] = ["专业", "简洁"]

    plan = IntentPlan(
        intent_summary=data.get("intent_summary", intent),
        content_style=data["content_style"],
        analysis_dimensions=data["analysis_dimensions"],
        narrative_structure=data["narrative_structure"],
        narrative_description=data.get("narrative_description", ""),
        persona=data.get("persona", "专业解说员"),
        persona_prompt=data.get("persona_prompt", "你是一位专业的视频解说员。"),
        tone_keywords=data["tone_keywords"],
        hook_strategy=data.get("hook_strategy", "用提问引起好奇心"),
        emotion_profile=data["emotion_profile"],
        fancy_text_guidance=data["fancy_text_guidance"],
        title_suggestion=data.get("title_suggestion", ""),
    )

    print(f"  Intent plan created:")
    print(f"    Summary: {plan.intent_summary}")
    print(f"    Style: {plan.content_style}")
    print(f"    Persona: {plan.persona}")
    print(f"    Structure: {plan.narrative_structure}")
    print(f"    Dimensions: {', '.join(plan.analysis_dimensions)}")
    print(f"    Title: {plan.title_suggestion}")
    return plan


# ---------------------------------------------------------------------------
# Step 3: Video analysis with intent focus
# ---------------------------------------------------------------------------

def _build_intent_vl_prompt(transcript_text, video_duration_str, intent_plan):
    """Build the analysis prompt for Qwen2.5-VL, tailored to the intent plan."""
    dimensions_str = "\n".join(
        f"  {i + 1}. {dim}" for i, dim in enumerate(intent_plan.analysis_dimensions)
    )
    return f"""你是一位{intent_plan.persona}。请分析这个视频（总时长约{video_duration_str}）。

## 你的任务

用户的创作意图是：**{intent_plan.intent_summary}**

你需要从视频中找出能体现以下维度的**具体片段**：
{dimensions_str}

## 重要：分析方法 vs 内容转述

你的任务是**识别和标注视频中体现特定技巧/方法/规律的片段**，而不是总结视频讲了什么内容。

举例说明区别：
- ❌ 内容转述："演讲者讲述了她创建AI伴侣的故事"
- ✅ 技巧分析："演讲者在02:30使用了个人悲伤故事作为情感锚点，这是一种经典的共情开场技巧"
- ❌ 内容转述："这段讲了职场中被领导批评的情节"
- ✅ 规律提炼："这段情节展示了'向上管理'的反面案例——在公开场合反驳领导"

对于每个片段，请回答：**这里体现了什么技巧/方法/规律？为什么这个片段是好的示范或反面案例？**

## 转录文本参考
{transcript_text}

## 输出格式

请返回严格的JSON格式（不要添加markdown围栏）：
{{
  "video_type": "视频类型",
  "overall_summary": "整体内容概要（2-3句）",
  "intent_findings": "与'{intent_plan.intent_summary}'直接相关的核心发现（3-5条，每条指出一个具体的技巧/方法/规律）",
  "events": [
    {{
      "start_time": "起始秒数",
      "end_time": "结束秒数",
      "description": "该片段体现了什么技巧/方法/规律（不是内容摘要！）",
      "technique_name": "技巧/方法的简短命名（如'情感锚点法'、'三段式递进'）",
      "visual_highlights": "画面中的关键细节（肢体语言、表情、道具等）",
      "emotional_tone": "情感基调描述",
      "emotion_tags": ["从以下选择: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm"],
      "importance": 8,
      "reason": "为什么这个片段适合用来讲解该技巧",
      "intent_relevance": "与'{intent_plan.intent_summary}'的关联度说明",
      "intent_score": 8
    }}
  ]
}}

注意：
- importance 评分 1-10，越高越适合做短视频素材
- intent_score 评分 1-10，越高越与创作意图相关
- emotion_tags 必须从这8个中选择: happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
- 时间戳必须是秒数（如 "123.5"），与视频实际时间对齐
- **description 必须说明技巧/方法，不能只复述视频内容**
- **technique_name 是必填的，每个片段都要命名所体现的技巧**
- 请尽量发现所有有价值的片段，宁多勿少"""


def analyze_video_with_intent(video_path, transcript, work_dir, intent_plan, vl_model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
    """Analyze video using Qwen2.5-VL with intent-focused prompting."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    analysis_path = os.path.join(work_dir, "intent_video_analysis.json")
    if os.path.exists(analysis_path):
        with open(analysis_path, "r", encoding="utf-8") as f:
            analysis = json.load(f)
        print(f"  Loaded cached intent analysis ({len(analysis.get('events', []))} events)")
        return analysis

    video_info = _get_video_info(video_path)
    duration_min = video_info["duration"] / 60
    video_duration_str = f"{duration_min:.1f}分钟"

    # Build transcript text (truncate if too long)
    transcript_lines = []
    for seg in transcript:
        t = f"[{seg['start']:.1f}s] {seg['text']}"
        transcript_lines.append(t)
    transcript_text = "\n".join(transcript_lines)
    if len(transcript_text) > 4000:
        transcript_text = transcript_text[:4000] + "\n... (转录文本已截断)"

    prompt = _build_intent_vl_prompt(transcript_text, video_duration_str, intent_plan)

    # Load model (4-bit quantized to fit L4 24GB with video frames)
    print(f"  Loading {vl_model_name}...")
    try:
        import flash_attn  # noqa: F401
        attn_impl = "flash_attention_2"
    except ImportError:
        attn_impl = "eager"
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
        print("  Using 4-bit quantization (saves ~10GB VRAM)")
    except ImportError:
        quantization_config = None
        print("  bitsandbytes not available, loading in bfloat16")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        vl_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        quantization_config=quantization_config,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(vl_model_name)

    # Extract frames as JPEG images with ffmpeg
    video_dur = video_info["duration"]
    nframes = min(int(video_dur * VL_FPS), VL_MAX_FRAMES)
    extract_fps = nframes / video_dur if video_dur > 0 else VL_FPS
    frames_dir = os.path.join(work_dir, "vl_frames")
    os.makedirs(frames_dir, exist_ok=True)
    existing_frames = sorted(
        f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")
    )
    if len(existing_frames) < nframes:
        print(f"  Extracting {nframes} frames from {video_dur:.0f}s video (ffmpeg)...")
        _run_ffmpeg(
            "-i", video_path,
            "-vf", f"fps={extract_fps:.6f},scale=480:-2",
            "-q:v", "2",
            os.path.join(frames_dir, "frame_%04d.jpg"),
        )
        existing_frames = sorted(
            f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".jpg")
        )
    print(f"  Using {len(existing_frames)} extracted frames")

    # Build frame paths and compute fps for temporal alignment
    frame_paths = [
        f"file://{os.path.abspath(os.path.join(frames_dir, f))}"
        for f in existing_frames
    ]
    frame_fps = len(existing_frames) / video_dur if video_dur > 0 else VL_FPS

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_paths,
                    "fps": frame_fps,
                    "total_pixels": VL_TOTAL_PIXELS,
                    "min_pixels": VL_MIN_PIXELS,
                    "max_pixels": VL_MAX_PIXELS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Process and generate
    print("  Analyzing video with intent focus (this may take a while)...")
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        **video_kwargs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=8192)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]

    # Unload VL model
    del model, processor
    _free_vram()
    print("  Qwen2.5-VL unloaded.")

    # Parse JSON response
    analysis = _parse_json_response(response)
    if analysis is None:
        raise RuntimeError(f"Failed to parse VL response:\n{response[:500]}")

    # Save
    with open(analysis_path, "w", encoding="utf-8") as f:
        json.dump(analysis, f, ensure_ascii=False, indent=2)
    print(f"  Found {len(analysis.get('events', []))} events.")
    return analysis


# ---------------------------------------------------------------------------
# Step 4: Intent-driven script generation
# ---------------------------------------------------------------------------

def _build_intent_script_prompt(analysis, intent_plan, target_duration, canvas_w, canvas_h):
    """Build script generation prompt tailored to the intent plan."""
    events = analysis.get("events", [])
    events = [e for e in events if e.get("importance", 0) >= MIN_IMPORTANCE]

    effects_desc = """可用花字特效列表（fancy_texts中使用）:
  - pop_zoom: 弹出缩放（文字从小弹大再回弹）
  - slide_in_left: 从左滑入
  - slide_in_right: 从右滑入
  - slide_in_top: 从上滑入
  - slide_in_bottom: 从下滑入
  - fade: 淡入淡出
  - word_highlight: 逐字高亮（每个字依次变色）
  - typewriter: 打字机效果（逐字出现）
  - scroll_marquee: 滚动字幕（从右到左滚过）
  - outlined: 描边文字
  - gradient_color: 渐变色文字
  - shadow_3d: 3D阴影文字
  - rotate_in: 旋转入场
  - bounce: 弹跳效果
  - multi_layer: 多层叠加（阴影+描边+主体）
  - emphasis_glow: 发光强调
  - shake: 抖动效果
  - subtitle_default: 默认字幕（底部居中）"""

    tone_str = "、".join(intent_plan.tone_keywords)

    # Build intent-specific narration guidance
    if intent_plan.content_style in ("analytical", "educational"):
        narration_guidance = f"""## 旁白写作的关键原则（极其重要！！）

你写的旁白是**对技巧/方法/规律的分析评论**，不是对视频内容的复述！

正确示范：
- "注意看，这里她用了一个非常高明的技巧——用自己最脆弱的经历来建立共情。这就是演讲中的'情感锚点法'"
- "她的语速在这里突然放慢了，配合3秒的停顿。这不是紧张，而是刻意为之——让观众有时间消化刚才的信息"
- "接下来这段很关键——她没有直接给出答案，而是用了三个递进的问题把观众的好奇心拉到最高点"

错误示范（不要这样写！）：
- "她讲述了自己失去朋友的故事" ← 这是内容复述
- "AI技术给人类带来了希望" ← 这是内容总结
- "她创建了一个AI伴侣来缅怀朋友" ← 这是情节概括

每段旁白应该回答：**视频此刻正在使用什么技巧？为什么有效？观众能从中学到什么？**

语气要求：{tone_str}（像一位{intent_plan.persona}在做专业点评）"""
    else:
        narration_guidance = f"""## 旁白写作指导

为每个片段编写中文旁白（口语化、有感染力、适合配音朗读）。
旁白应紧扣用户意图"{intent_plan.intent_summary}"，而非简单转述视频内容。
语气要求：{tone_str}"""

    return f"""你是{intent_plan.persona_prompt}

内容风格：{intent_plan.content_style}
叙事结构：{intent_plan.narrative_description}
开头策略：{intent_plan.hook_strategy}

以下是一个{analysis.get('video_type', '未知')}类型视频的关键事件分析：

整体概要：{analysis.get('overall_summary', '')}
意图相关发现：{analysis.get('intent_findings', '')}

事件列表：
{json.dumps(events, ensure_ascii=False, indent=2)}

请从中选取最精华的事件，生成一个约{target_duration}秒的短视频解说脚本。

{narration_guidance}

## 花字文案写作指导

花字（fancy_texts）是叠加在画面上的文字特效，用来**强调关键信息**。

花字应该写什么：
- 技巧/方法的名称（如"情感锚点法"、"三段式递进"）
- 关键数据或数字
- 核心观点的精炼表达（6-12个字）
- 观众能记住的金句

花字**不要**写什么：
- 不要把整段旁白都放到花字里
- 不要写太长的句子（超过15个字就太长了）
- 花字必须是**中文**

每个片段至少1个花字，重点片段2-3个。花字 timing 的起止时间是**相对于该片段开始的秒数**。

## 情感向量说明
格式：[happy, angry, sad, afraid, disgusted, melancholic, surprised, calm]
- 紧张悬疑 → 提高 afraid/surprised
- 感人段落 → 提高 sad/melancholic
- 激昂振奋 → 提高 happy
- 平静叙述 → 提高 calm
- 向量总和建议不超过0.8

## 可用特效菜单

转场(transition): {', '.join(TRANSITIONS)}
调色(color_grade): {', '.join(COLOR_GRADES)}
可选特效(effects):
  - slow_motion: 指定时间范围和倍率（如0.5=半速）
  - ken_burns: 缩放平移（适合静态画面）
标题卡片(title_card): 段落间2-3秒的文字过渡卡（可选）

## 花字特效类型
{effects_desc}

画布尺寸：{canvas_w}x{canvas_h}
花字密度建议：{intent_plan.fancy_text_guidance.get('density', 'medium')}
花字风格建议：{intent_plan.fancy_text_guidance.get('style_tone', 'clean')}

## 要求
1. 总旁白时长控制在{target_duration}秒左右
2. 开头必须有hook（{intent_plan.hook_strategy}）
3. 结尾有总结或金句
4. 按叙事逻辑排列，确保连贯性
5. 花字文案必须是中文，简洁有力（6-12字），突出技巧名称或核心观点
6. 每个片段必须包含旁白字幕（effect=subtitle_default），位置在画面底部

请返回严格的JSON数组（不要添加markdown围栏）：
[
  {{
    "segment_id": 0,
    "start_time": 123.5,
    "end_time": 245.8,
    "narration": "注意看这里，她用了一个非常高明的技巧...",
    "emo_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
    "transition": "fade",
    "effects": {{
      "color_grade": "cinematic_warm"
    }},
    "title_card": null,
    "fancy_texts": [
      {{
        "text": "情感锚点法",
        "effect": "pop_zoom",
        "position": [{canvas_w // 2}, {canvas_h // 3}],
        "timing": [1.0, 4.0],
        "size": 72,
        "color": "#FF6B6B",
        "layer": 1
      }},
      {{
        "text": "旁白字幕文字（和narration一致）",
        "effect": "subtitle_default",
        "position": [{canvas_w // 2}, {canvas_h - 200}],
        "timing": [0.0, 8.0],
        "size": 44,
        "color": "#FFFFFF",
        "layer": 0
      }}
    ]
  }}
]"""


def generate_intent_script(analysis, intent_plan, llm_client, target_duration, canvas_w, canvas_h):
    """Generate narration script with fancy text choreography, driven by intent plan."""
    prompt = _build_intent_script_prompt(analysis, intent_plan, target_duration, canvas_w, canvas_h)

    print("  Generating intent-driven script...")
    response = llm_client.chat(prompt, temperature=0.7)

    script = _parse_json_response(response)
    if script is None:
        raise RuntimeError(f"Failed to parse LLM script response:\n{response[:500]}")

    if isinstance(script, dict):
        script = script.get("segments", script.get("script", []))

    # Validate and clean up each item
    cleaned = []
    for item in script:
        cleaned_item = {
            "segment_id": item.get("segment_id", len(cleaned)),
            "start_time": float(item.get("start_time", 0)),
            "end_time": float(item.get("end_time", 0)),
            "narration": item.get("narration", ""),
            "emo_vector": item.get("emo_vector", [0, 0, 0, 0, 0, 0, 0, 0.5]),
            "transition": item.get("transition", "fade"),
            "effects": item.get("effects", {}),
            "title_card": item.get("title_card"),
            "fancy_texts": item.get("fancy_texts", []),
        }
        # Validate transition
        if cleaned_item["transition"] not in TRANSITIONS:
            cleaned_item["transition"] = "fade"
        # Validate emo_vector length
        vec = cleaned_item["emo_vector"]
        if len(vec) != 8:
            cleaned_item["emo_vector"] = [0, 0, 0, 0, 0, 0, 0, 0.5]
        # Clamp emo_vector values to [0, 1]
        cleaned_item["emo_vector"] = [max(0.0, min(1.0, float(v))) for v in cleaned_item["emo_vector"]]

        # Validate fancy_texts
        seg_duration = cleaned_item["end_time"] - cleaned_item["start_time"]
        validated_fancy = []
        if isinstance(cleaned_item["fancy_texts"], list):
            for ft in cleaned_item["fancy_texts"]:
                if not isinstance(ft, dict):
                    continue
                vft = {
                    "text": str(ft.get("text", "")),
                    "effect": ft.get("effect", "fade"),
                    "position": ft.get("position", [canvas_w // 2, canvas_h // 2]),
                    "timing": ft.get("timing", [0, seg_duration]),
                    "size": ft.get("size", 60),
                    "color": ft.get("color", "#FFFFFF"),
                    "layer": ft.get("layer", 0),
                }
                # Validate effect
                if vft["effect"] not in FANCY_TEXT_EFFECTS:
                    vft["effect"] = "fade"
                # Validate position
                if not isinstance(vft["position"], list) or len(vft["position"]) < 2:
                    vft["position"] = [canvas_w // 2, canvas_h // 2]
                vft["position"] = [
                    max(0, min(canvas_w, int(vft["position"][0]))),
                    max(0, min(canvas_h, int(vft["position"][1]))),
                ]
                # Validate timing
                if not isinstance(vft["timing"], list) or len(vft["timing"]) < 2:
                    vft["timing"] = [0, seg_duration]
                vft["timing"] = [float(vft["timing"][0]), float(vft["timing"][1])]
                # Validate size
                vft["size"] = int(vft.get("size", 60)) if vft.get("size") else 60
                # Validate layer
                vft["layer"] = int(vft.get("layer", 0)) if vft.get("layer") is not None else 0
                validated_fancy.append(vft)
        cleaned_item["fancy_texts"] = validated_fancy
        cleaned.append(cleaned_item)

    print(f"  Script generated: {len(cleaned)} segments")
    for i, item in enumerate(cleaned):
        dur = item["end_time"] - item["start_time"]
        n_fancy = len(item["fancy_texts"])
        print(f"    [{i}] {item['start_time']:.1f}-{item['end_time']:.1f}s "
              f"({dur:.1f}s) transition={item['transition']} "
              f"fancy={n_fancy} | {item['narration'][:40]}...")

    return cleaned


# ---------------------------------------------------------------------------
# Step 5: Narration with emotion profile
# ---------------------------------------------------------------------------

def _apply_emotion_profile(emo_vector, emotion_profile):
    """Blend segment emotion vector with the intent's base emotion profile.

    When allow_high_emotion is False (analytical/educational content),
    the base vector dominates (80%) and individual dims are capped at 0.4
    to ensure calm, even narration. When True (emotional/entertaining),
    the segment vector leads (70%) with a 0.7 cap.
    """
    base = emotion_profile.get("base_vector", [0] * 8)
    allow_high = emotion_profile.get("allow_high_emotion", False)
    if allow_high:
        # Emotional/entertaining: segment leads
        blended = [emo_vector[i] * 0.7 + base[i] * 0.3 for i in range(8)]
        cap = 0.7
    else:
        # Analytical/educational: base dominates, keep it calm
        blended = [emo_vector[i] * 0.2 + base[i] * 0.8 for i in range(8)]
        cap = 0.4
    blended = [min(v, cap) for v in blended]
    return [max(0.0, min(1.0, v)) for v in blended]


def generate_narration_with_profile(script, ref_audio, work_dir, intent_plan, model_dir="checkpoints", use_fp16=True):
    """Generate narration audio with emotion profile biasing for each script segment."""
    tts_dir = os.path.join(work_dir, "tts_output")
    os.makedirs(tts_dir, exist_ok=True)

    # Check if all already generated
    all_exist = all(
        os.path.exists(os.path.join(tts_dir, f"narration_{i:04d}.wav"))
        for i in range(len(script))
    )
    if all_exist:
        print("  All narration files already exist, loading durations...")
        for i, item in enumerate(script):
            wav_path = os.path.join(tts_dir, f"narration_{i:04d}.wav")
            item["wav_path"] = wav_path
            item["wav_duration"] = get_audio_duration(wav_path)
        return script

    print("  Loading IndexTTS2...")
    tts = _init_tts(model_dir, use_fp16)

    for i, item in enumerate(script):
        output_path = os.path.join(tts_dir, f"narration_{i:04d}.wav")

        if os.path.exists(output_path):
            item["wav_path"] = output_path
            item["wav_duration"] = get_audio_duration(output_path)
            print(f"  [{i}] Cached: {item['wav_duration']:.1f}s")
            continue

        narration = item["narration"]
        emo_vector = item.get("emo_vector", [0, 0, 0, 0, 0, 0, 0, 0.5])

        # Apply emotion profile biasing
        emo_vector = _apply_emotion_profile(emo_vector, intent_plan.emotion_profile)

        # Normalize emotion vector
        if any(v > 0 for v in emo_vector):
            emo_vector = tts.normalize_emo_vec(emo_vector)
        else:
            emo_vector = None

        try:
            tts.infer(
                spk_audio_prompt=ref_audio,
                text=narration,
                output_path=output_path,
                emo_vector=emo_vector,
                verbose=False,
            )
        except Exception as e:
            print(f"  [{i}] TTS failed ({e}), using silence placeholder")
            silence_dur = max(2.0, len(narration) * 0.15)
            silence = np.zeros(int(silence_dur * TTS_SAMPLE_RATE), dtype=np.float32)
            sf.write(output_path, silence, TTS_SAMPLE_RATE)

        item["wav_path"] = output_path
        item["wav_duration"] = get_audio_duration(output_path)
        print(f"  [{i}] Generated: {item['wav_duration']:.1f}s | "
              f"emo={[f'{v:.2f}' for v in (emo_vector or [])]} | "
              f"{narration[:30]}...")

    # Unload TTS
    del tts
    _free_vram()
    print("  IndexTTS2 unloaded.")
    return script


# ---------------------------------------------------------------------------
# Step 6a: Canvas preparation
# ---------------------------------------------------------------------------

def prepare_canvas(video_path, work_dir, orientation="portrait"):
    """Convert video to target orientation with blurred background fill."""
    canvas_w, canvas_h = ORIENTATIONS.get(orientation, ORIENTATIONS["portrait"])
    info = _get_video_info(video_path)
    w, h = info["width"], info["height"]

    canvas_path = os.path.join(work_dir, "canvas_source.mp4")
    if os.path.exists(canvas_path):
        return canvas_path

    target_aspect = canvas_w / canvas_h
    source_aspect = w / h

    if (orientation == "portrait" and source_aspect > 1.0) or \
       (orientation == "landscape" and source_aspect < 1.0):
        # Aspect mismatch: use blurred background fill
        _run_ffmpeg(
            "-i", video_path,
            "-filter_complex",
            f"[0:v]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=increase,"
            f"crop={canvas_w}:{canvas_h},gblur=sigma=30[bg];"
            f"[0:v]scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2",
            "-c:a", "copy",
            canvas_path,
        )
    else:
        # Similar aspect: scale and pad
        _run_ffmpeg(
            "-i", video_path,
            "-vf", f"scale={canvas_w}:{canvas_h}:force_original_aspect_ratio=decrease,"
                   f"pad={canvas_w}:{canvas_h}:(ow-iw)/2:(oh-ih)/2:black",
            "-c:a", "copy",
            canvas_path,
        )

    print(f"  Canvas prepared: {w}x{h} -> {canvas_w}x{canvas_h} ({orientation})")
    return canvas_path


# ---------------------------------------------------------------------------
# Step 6b: Cut segments with canvas dimensions
# ---------------------------------------------------------------------------

def cut_and_apply_effects_canvas(canvas_path, script, work_dir, canvas_w, canvas_h):
    """Cut segments from canvas video and apply per-segment effects."""
    segments_dir = os.path.join(work_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)

    for i, item in enumerate(script):
        seg_path = os.path.join(segments_dir, f"seg_{i:04d}.mp4")
        if os.path.exists(seg_path):
            item["seg_path"] = seg_path
            continue

        start = item["start_time"]
        end = item["end_time"]
        effects = item.get("effects", {})

        # Build video filter chain
        vfilters = []

        # Color grading
        color_grade = effects.get("color_grade", "none")
        grade_filter = _get_color_grade_filter(color_grade)
        if grade_filter != "null":
            vfilters.append(grade_filter)

        # Ken Burns zoom+pan
        if effects.get("ken_burns"):
            vfilters.append(
                f"zoompan=z='min(zoom+0.001,1.3)':d=125:"
                f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":s={canvas_w}x{canvas_h}:fps=30"
            )

        # Slow motion
        slow = effects.get("slow_motion")
        if slow:
            factor = slow.get("factor", 0.5)
            vfilters.append(f"setpts={1 / factor}*PTS")

        vfilter_str = ",".join(vfilters) if vfilters else "null"

        # Audio filter for slow motion
        afilter = "anull"
        if slow:
            factor = slow.get("factor", 0.5)
            atempo_val = max(0.5, min(2.0, factor))
            afilter = f"atempo={atempo_val}"

        duration = end - start
        _run_ffmpeg(
            "-ss", str(start),
            "-i", canvas_path,
            "-t", str(duration),
            "-vf", vfilter_str,
            "-af", afilter,
            *X264_ARGS, *AAC_ARGS,
            seg_path,
        )
        item["seg_path"] = seg_path
        item["seg_duration"] = _get_video_info(seg_path)["duration"]
        print(f"  [{i}] Cut {start:.1f}-{end:.1f}s ({duration:.1f}s) grade={color_grade}")

    return script


# ---------------------------------------------------------------------------
# Fancy text ASS rendering
# ---------------------------------------------------------------------------

def _hex_to_ass_color(hex_color):
    """Convert hex color (#RRGGBB) to ASS color (&HBBGGRR&).

    Also accepts named colors (white, cyan, red, etc.).
    Returns &HFFFFFF& on failure.
    """
    if isinstance(hex_color, str):
        # Check named colors
        lower = hex_color.lower().strip()
        if lower in _NAMED_COLORS:
            return _NAMED_COLORS[lower]
        # Strip "#" if present
        hex_color = hex_color.strip().lstrip("#")
        if len(hex_color) == 6:
            try:
                r = hex_color[0:2]
                g = hex_color[2:4]
                b = hex_color[4:6]
                return f"&H{b.upper()}{g.upper()}{r.upper()}&"
            except (ValueError, IndexError):
                pass
    return "&HFFFFFF&"


def _ensure_cjk_font():
    """Ensure a CJK font is available on the system. Returns the font name to use in ASS."""
    import shutil
    # Check if Noto Sans SC is already installed
    fc_list = shutil.which("fc-list")
    if fc_list:
        try:
            result = subprocess.run(
                ["fc-list", ":lang=zh", "family"],
                capture_output=True, text=True, timeout=5,
            )
            families = result.stdout.strip()
            if "Noto Sans CJK" in families or "Noto Sans SC" in families:
                return "Noto Sans SC"
            if "WenQuanYi" in families:
                return "WenQuanYi Micro Hei"
        except Exception:
            pass
    # Try to install CJK font (works on Colab/Ubuntu)
    try:
        subprocess.run(
            ["apt-get", "install", "-y", "-qq", "fonts-noto-cjk"],
            capture_output=True, timeout=60,
        )
        # Refresh font cache
        subprocess.run(["fc-cache", "-f"], capture_output=True, timeout=30)
        print("  Installed fonts-noto-cjk for Chinese text rendering")
        return "Noto Sans CJK SC"
    except Exception:
        pass
    return "Noto Sans SC"  # Fallback, hope for the best


# Module-level font detection (runs once at import time)
_CJK_FONT = None


def _get_cjk_font():
    global _CJK_FONT
    if _CJK_FONT is None:
        _CJK_FONT = _ensure_cjk_font()
    return _CJK_FONT


def _generate_fancy_ass_header(canvas_w, canvas_h):
    """Generate ASS file header with fancy styles."""
    font = _get_cjk_font()
    return f"""[Script Info]
Title: Intent Video Fancy Text
ScriptType: v4.00+
PlayResX: {canvas_w}
PlayResY: {canvas_h}
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: FancyDefault,{font},52,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,2,30,30,120,1
Style: FancyOutline,{font},60,&H00FFFFFF,&H000000FF,&H004040FF,&H00000000,-1,0,0,0,100,100,0,0,1,6,0,5,30,30,120,1
Style: FancyGlow,{font},64,&H0000FFFF,&H000000FF,&H0000FFFF,&H00000000,-1,0,0,0,100,100,0,0,1,0,4,5,30,30,120,1
Style: FancyTitle,{font},80,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,5,30,30,400,1
Style: FancyShadow,{font},64,&H00FFFFFF,&H000000FF,&H00333333,&H00666666,-1,0,0,0,100,100,0,0,1,3,4,5,30,30,120,1
Style: SubtitleNarr,{font},48,&H00FFFFFF,&H000000FF,&H00000000,&H96000000,-1,0,0,0,100,100,0,0,1,3,1,2,30,30,100,1
Style: ProgressBar,Arial,10,&H0000BFFF,&H000000FF,&H0000BFFF,&H0000BFFF,0,0,0,0,100,100,0,0,3,0,0,7,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def render_fancy_text(fancy_item, segment_offset, canvas_w, canvas_h):
    """Render a single fancy text item as ASS Dialogue lines.

    Returns a list of ASS Dialogue line strings (each ending with newline).
    """
    text = fancy_item.get("text", "")
    effect = fancy_item.get("effect", "fade")
    pos = fancy_item.get("position", [canvas_w // 2, canvas_h // 2])
    timing = fancy_item.get("timing", [0, 3])
    size = int(fancy_item.get("size", 60))
    color = fancy_item.get("color", "#FFFFFF")
    layer = int(fancy_item.get("layer", 0))

    x, y = int(pos[0]), int(pos[1])
    abs_start = segment_offset + timing[0]
    abs_end = segment_offset + timing[1]
    dur_sec = abs_end - abs_start
    dur_ms = int(dur_sec * 1000)

    ass_color = _hex_to_ass_color(color)
    # Strip the &H prefix and & suffix for inline use
    ass_color_inner = ass_color[2:-1] if ass_color.startswith("&H") and ass_color.endswith("&") else "FFFFFF"

    lines = []

    def _dl(l, s, e, tags, txt, style="FancyDefault"):
        """Helper to format a Dialogue line."""
        return (
            f"Dialogue: {l},{_format_ass_time(s)},{_format_ass_time(e)},"
            f"{style},,0,0,0,,{tags}{txt}\n"
        )

    if effect == "pop_zoom":
        tags = (
            f"{{\\an5\\pos({x},{y})\\fs{size}\\c{ass_color}"
            f"\\fscx20\\fscy20"
            f"\\t(0,200,\\fscx115\\fscy115)"
            f"\\t(200,350,\\fscx100\\fscy100)"
            f"\\fad(0,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "slide_in_left":
        tags = (
            f"{{\\an5\\move(-300,{y},{x},{y},0,400)"
            f"\\fs{size}\\c{ass_color}\\fad(0,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "slide_in_right":
        tags = (
            f"{{\\an5\\move({canvas_w + 300},{y},{x},{y},0,400)"
            f"\\fs{size}\\c{ass_color}\\fad(0,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "slide_in_top":
        tags = (
            f"{{\\an5\\move({x},-100,{x},{y},0,400)"
            f"\\fs{size}\\c{ass_color}\\fad(0,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "slide_in_bottom":
        tags = (
            f"{{\\an5\\move({x},{canvas_h + 100},{x},{y},0,400)"
            f"\\fs{size}\\c{ass_color}\\fad(0,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "fade":
        tags = f"{{\\an5\\pos({x},{y})\\fs{size}\\c{ass_color}\\fad(400,400)}}"
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "word_highlight":
        chars = list(text)
        if not chars:
            return lines
        char_dur = dur_sec / len(chars)
        # Determine highlight color (use specified color for highlight, white for normal)
        highlight_color = ass_color_inner
        for j in range(len(chars)):
            before = text[:j]
            current = chars[j]
            after = text[j + 1:]
            highlighted = f"{before}{{\\c&H{highlight_color}&}}{current}{{\\c&HFFFFFF&}}{after}"
            c_start = abs_start + j * char_dur
            c_end = abs_start + (j + 1) * char_dur
            tags = f"{{\\an5\\pos({x},{y})\\fs{size}}}"
            lines.append(_dl(layer, c_start, c_end, tags, highlighted))

    elif effect == "typewriter":
        chars = list(text)
        if not chars:
            return lines
        char_dur = dur_sec / len(chars)
        for j in range(len(chars)):
            visible = text[:j + 1]
            c_start = abs_start + j * char_dur
            c_end = abs_start + (j + 1) * char_dur if j < len(chars) - 1 else abs_end
            tags = f"{{\\an5\\pos({x},{y})\\fs{size}\\c{ass_color}}}"
            lines.append(_dl(layer, c_start, c_end, tags, visible))

    elif effect == "scroll_marquee":
        tags = (
            f"{{\\an5\\move({canvas_w + 300},{y},{-300},{y},0,{dur_ms})"
            f"\\fs{size}\\c{ass_color}}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "outlined":
        tags = (
            f"{{\\an5\\pos({x},{y})\\bord8\\3c&H{ass_color_inner}&"
            f"\\fs{size}\\c{ass_color}\\fad(300,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "gradient_color":
        # Top half: specified color
        top_tags = (
            f"{{\\an5\\pos({x},{y})\\clip(0,0,{canvas_w},{y})"
            f"\\c&H{ass_color_inner}&\\fs{size}\\fad(300,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, top_tags, text))
        # Bottom half: lighter shade (brighten by 40 per channel, or white)
        bottom_tags = (
            f"{{\\an5\\pos({x},{y})\\clip(0,{y},{canvas_w},{canvas_h})"
            f"\\c&HFFFFFF&\\fs{size}\\fad(300,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, bottom_tags, text))

    elif effect == "shadow_3d":
        # Shadow layer
        shadow_tags = (
            f"{{\\an5\\pos({x + 4},{y + 4})\\c&H333333&\\alpha&H60&\\fs{size}}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, shadow_tags, text))
        # Main layer
        main_tags = f"{{\\an5\\pos({x},{y})\\shad0\\fs{size}\\c{ass_color}}}"
        lines.append(_dl(layer, abs_start, abs_end, main_tags, text))

    elif effect == "rotate_in":
        tags = (
            f"{{\\an5\\pos({x},{y})\\frz30\\fscx0\\fscy0"
            f"\\t(0,400,\\frz0\\fscx100\\fscy100)"
            f"\\fs{size}\\c{ass_color}\\fad(0,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "bounce":
        tags = (
            f"{{\\an5\\pos({x},{y})\\fscx100\\fscy100"
            f"\\t(0,150,\\fscy125)\\t(150,300,\\fscy100)"
            f"\\t(300,400,\\fscy108)\\t(400,500,\\fscy100)"
            f"\\fs{size}\\c{ass_color}\\fad(0,300)}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    elif effect == "multi_layer":
        # Shadow
        shadow_tags = (
            f"{{\\an5\\pos({x + 3},{y + 3})\\c&H333333&\\alpha&H80&\\fs{size}}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, shadow_tags, text))
        # Color outline
        outline_tags = (
            f"{{\\an5\\pos({x + 1},{y + 1})\\c&H{ass_color_inner}&\\alpha&H40&\\fs{size}}}"
        )
        lines.append(_dl(layer + 1, abs_start, abs_end, outline_tags, text))
        # Main
        main_tags = f"{{\\an5\\pos({x},{y})\\fs{size}\\c{ass_color}}}"
        lines.append(_dl(layer + 2, abs_start, abs_end, main_tags, text))

    elif effect == "emphasis_glow":
        # Glow layer
        glow_tags = (
            f"{{\\an5\\pos({x},{y})\\blur6\\alpha&H50&\\c&H{ass_color_inner}&"
            f"\\t(0,600,\\blur3)\\t(600,1200,\\blur6)\\fs{size + 8}}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, glow_tags, text))
        # Main layer
        main_tags = f"{{\\an5\\pos({x},{y})\\blur0\\fs{size}\\c{ass_color}}}"
        lines.append(_dl(layer, abs_start, abs_end, main_tags, text))

    elif effect == "shake":
        # Generate ~15 short dialogue lines with random jitter
        n_shakes = 15
        shake_dur = dur_sec / n_shakes
        for j in range(n_shakes):
            jx = x + random.randint(-5, 5)
            jy = y + random.randint(-5, 5)
            s_start = abs_start + j * shake_dur
            s_end = abs_start + (j + 1) * shake_dur if j < n_shakes - 1 else abs_end
            tags = f"{{\\an5\\pos({jx},{jy})\\fs{size}\\c{ass_color}}}"
            lines.append(_dl(layer, s_start, s_end, tags, text))

    elif effect == "subtitle_default":
        tags = (
            f"{{\\an2\\pos({canvas_w // 2},{canvas_h - 200})"
            f"\\fad(200,200)\\fs{size}\\c{ass_color}}}"
        )
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    else:
        # Fallback to fade
        tags = f"{{\\an5\\pos({x},{y})\\fs{size}\\c{ass_color}\\fad(400,400)}}"
        lines.append(_dl(layer, abs_start, abs_end, tags, text))

    return lines


def generate_fancy_ass(script, total_duration, work_dir, canvas_w, canvas_h):
    """Generate ASS subtitle file with fancy text effects."""
    ass_path = os.path.join(work_dir, "fancy_subtitles.ass")
    lines = [_generate_fancy_ass_header(canvas_w, canvas_h)]
    offsets = _compute_timeline(script)

    for i, item in enumerate(script):
        segment_offset = offsets[i]
        fancy_texts = item.get("fancy_texts", [])
        for ft in fancy_texts:
            rendered = render_fancy_text(ft, segment_offset, canvas_w, canvas_h)
            lines.extend(rendered)

    # Progress bar at bottom
    lines.append(
        f"Dialogue: 1,{_format_ass_time(0)},{_format_ass_time(total_duration)},"
        f"ProgressBar,,0,0,0,,{{\\p1\\clip(0,{canvas_h - 10},0,{canvas_h})"
        f"\\t(0,{int(total_duration * 1000)},\\clip(0,{canvas_h - 10},{canvas_w},{canvas_h}))}}"
        f"m 0 0 l {canvas_w} 0 l {canvas_w} 10 l 0 10{{\\p0}}\n"
    )

    with open(ass_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"  Generated fancy ASS subtitles: {ass_path}")
    return ass_path


# ---------------------------------------------------------------------------
# Step 6f: Final mix with fancy text
# ---------------------------------------------------------------------------

def _format_srt_time(seconds):
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    ms = int((s - int(s)) * 1000)
    return f"{h:02d}:{m:02d}:{int(s):02d},{ms:03d}"


def _generate_srt(script, work_dir):
    """Generate SRT subtitle file."""
    srt_path = os.path.join(work_dir, "subtitles.srt")
    offsets = _compute_timeline(script)

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(script):
            start = offsets[i]
            end = start + item.get("wav_duration", 5.0)
            f.write(f"{i + 1}\n")
            f.write(f"{_format_srt_time(start)} --> {_format_srt_time(end)}\n")
            f.write(f"{item['narration']}\n\n")

    print(f"  Generated SRT: {srt_path}")
    return srt_path


def final_mix_with_fancy(concat_path, script, audio_path, work_dir, output_path, canvas_w, canvas_h):
    """Mix narration + original audio, burn fancy subtitles, output final video."""
    import shutil

    # Build narration audio track
    narration_track = os.path.join(work_dir, "narration_full.wav")
    if not os.path.exists(narration_track):
        _build_narration_track(script, concat_path, narration_track)

    # Get total duration
    total_duration = _get_video_info(concat_path)["duration"]

    # Generate fancy ASS subtitles
    ass_path = generate_fancy_ass(script, total_duration, work_dir, canvas_w, canvas_h)

    # Generate SRT (standalone)
    srt_output = str(Path(output_path).with_suffix(".srt"))
    _generate_srt(script, work_dir)
    shutil.copy2(os.path.join(work_dir, "subtitles.srt"), srt_output)

    # Final mix: video + narration (full vol) + original audio (low vol) + subtitles
    _run_ffmpeg(
        "-i", concat_path,
        "-i", narration_track,
        "-filter_complex",
        f"[0:a]volume={ORIGINAL_VOLUME}[bg];"
        f"[1:a]volume={NARRATION_VOLUME}[speech];"
        f"[speech][bg]amix=inputs=2:duration=longest:normalize=0[aout];"
        f"[0:v]ass='{ass_path.replace(':', '\\:').replace(chr(92), '/')}'[vout]",
        "-map", "[vout]", "-map", "[aout]",
        *X264_ARGS, "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    )
    print(f"  Final output: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def create_intent_video(
    video_path,
    ref_audio,
    intent,
    output_path=None,
    orientation="portrait",
    work_dir="intent_workspace",
    llm_api_key=None,
    llm_api_base="https://api.openai.com/v1",
    llm_model="gpt-4o-mini",
    vl_model="Qwen/Qwen2.5-VL-7B-Instruct",
    model_dir="checkpoints",
    target_duration=180,
    use_fp16=True,
    cleanup=False,
):
    """
    Main pipeline: create an intent-driven video from a source video.

    Args:
        video_path: Input video file path.
        ref_audio: Reference audio for TTS voice cloning.
        intent: Natural language intent description.
        output_path: Output video path. Defaults to {input}_intent.mp4.
        orientation: Output orientation ("portrait" or "landscape").
        work_dir: Directory for intermediate files.
        llm_api_key: API key for OpenAI-compatible LLM.
        llm_api_base: Base URL for the LLM API.
        llm_model: Model name for script generation.
        vl_model: Qwen2.5-VL model name/path.
        model_dir: IndexTTS2 checkpoint directory.
        target_duration: Target output duration in seconds.
        use_fp16: Use FP16 for IndexTTS2 inference.
        cleanup: Delete intermediate files after completion.
    """
    video_path = str(video_path)
    ref_audio = str(ref_audio)
    if output_path is None:
        stem = Path(video_path).stem
        output_path = str(Path(video_path).parent / f"{stem}_intent.mp4")

    canvas_w, canvas_h = ORIENTATIONS.get(orientation, ORIENTATIONS["portrait"])

    video_work_dir = os.path.join(work_dir, Path(video_path).stem)
    os.makedirs(video_work_dir, exist_ok=True)

    # Detect ffmpeg version
    try:
        r = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        m = re.search(r"ffmpeg version (\d+)\.(\d+)", r.stdout)
        ffmpeg_ver = f"{m.group(1)}.{m.group(2)}" if m else "unknown"
    except Exception:
        ffmpeg_ver = "unknown"

    print(f"\n{'=' * 60}")
    print(f"Intent-Driven Video Pipeline")
    print(f"Input:       {video_path}")
    print(f"Ref audio:   {ref_audio}")
    print(f"Output:      {output_path}")
    print(f"Work dir:    {video_work_dir}")
    print(f"Intent:      {intent}")
    print(f"Orientation: {orientation} ({canvas_w}x{canvas_h})")
    print(f"ffmpeg:      {ffmpeg_ver}")
    print(f"Target:      {target_duration}s ({target_duration / 60:.1f}min)")
    print(f"{'=' * 60}")

    # Load checkpoint (use separate checkpoint file)
    ckpt_path = os.path.join(video_work_dir, INTENT_CHECKPOINT_FILE)
    try:
        with open(ckpt_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)
    except FileNotFoundError:
        ckpt = None
    done = ckpt.get("step", 0) if ckpt else 0
    state = ckpt or {}

    def _save_intent_checkpoint(step, **data):
        """Save pipeline progress with intent-specific checkpoint file."""
        data.pop("step", None)
        data["step"] = step
        with open(ckpt_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Checkpoint saved: step {step}")

    def _clear_intent_checkpoint():
        try:
            os.remove(ckpt_path)
        except FileNotFoundError:
            pass

    # --- Step 0: Intent Planning ---
    if done < 1:
        print("\n[Step 0/6] Planning intent...")
        llm_client = LLMClient(api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
        intent_plan = plan_intent(intent, llm_client)
        state["intent_plan"] = dataclasses.asdict(intent_plan)
        _save_intent_checkpoint(1, **state)
    else:
        intent_plan = IntentPlan(**state["intent_plan"])
        print(f"[Step 0/6] Skipped (cached: {intent_plan.intent_summary})")

    # --- Step 1: Extract audio ---
    if done < 2:
        print("\n[Step 1/6] Extracting audio...")
        audio_path = extract_audio(video_path, video_work_dir)
        state["audio_path"] = audio_path
        _save_intent_checkpoint(2, **state)
    else:
        audio_path = state["audio_path"]
        print("[Step 1/6] Skipped (cached)")

    # --- Step 2: Transcription ---
    if done < 3:
        print("\n[Step 2/6] Transcribing audio...")
        transcript = transcribe_audio(audio_path, video_work_dir)
        state["transcript"] = transcript
        _save_intent_checkpoint(3, **state)
    else:
        transcript = state.get("transcript", [])
        print(f"[Step 2/6] Skipped (cached, {len(transcript)} segments)")

    # --- Step 3: Video analysis with intent ---
    if done < 4:
        print("\n[Step 3/6] Analyzing video with intent focus...")
        analysis = analyze_video_with_intent(video_path, transcript, video_work_dir, intent_plan, vl_model)
        state["analysis"] = analysis
        _save_intent_checkpoint(4, **state)
    else:
        analysis = state.get("analysis", {})
        print(f"[Step 3/6] Skipped (cached, {len(analysis.get('events', []))} events)")

    # --- Step 4: Script generation ---
    if done < 5:
        print("\n[Step 4/6] Generating intent-driven script and effects...")
        llm_client = LLMClient(api_key=llm_api_key, api_base=llm_api_base, model=llm_model)
        script = generate_intent_script(analysis, intent_plan, llm_client, target_duration, canvas_w, canvas_h)
        script_path = os.path.join(video_work_dir, "intent_script.json")
        with open(script_path, "w", encoding="utf-8") as f:
            json.dump(script, f, ensure_ascii=False, indent=2)
        state["script"] = script
        _save_intent_checkpoint(5, **state)
    else:
        script = state.get("script", [])
        print(f"[Step 4/6] Skipped (cached, {len(script)} segments)")

    # --- Step 5: TTS narration with emotion profile ---
    if done < 6:
        print("\n[Step 5/6] Generating narration with emotion profile...")
        script = generate_narration_with_profile(script, ref_audio, video_work_dir, intent_plan, model_dir, use_fp16)
        state["script"] = script
        _save_intent_checkpoint(6, **state)
    else:
        script = state.get("script", [])
        print(f"[Step 5/6] Skipped (cached)")

    # --- Step 6: Video assembly ---
    print("\n[Step 6/6] Assembling intent video...")

    # 6a. Canvas preparation
    print(f"  6a. Preparing {orientation} canvas...")
    canvas_path = prepare_canvas(video_path, video_work_dir, orientation)

    # 6b. Cut segments + effects
    print("  6b. Cutting segments and applying effects...")
    script = cut_and_apply_effects_canvas(canvas_path, script, video_work_dir, canvas_w, canvas_h)

    # 6c. Title cards
    print("  6c. Generating title cards...")
    script = generate_title_cards(script, video_work_dir)

    # 6d. Concatenate with transitions
    print("  6d. Concatenating with transitions...")
    concat_path = concat_with_transitions(script, video_work_dir)

    # 6f. Final mix with fancy text
    print("  6f. Final mix with fancy text + audio...")
    final_mix_with_fancy(concat_path, script, audio_path, video_work_dir, output_path, canvas_w, canvas_h)

    _clear_intent_checkpoint()

    # Summary
    output_info = _get_video_info(output_path)
    print(f"\n{'=' * 60}")
    print(f"Done! Output: {output_path}")
    print(f"Duration: {output_info['duration']:.1f}s ({output_info['duration'] / 60:.1f}min)")
    print(f"Resolution: {output_info['width']}x{output_info['height']}")
    print(f"Intent: {intent_plan.intent_summary}")
    print(f"Title: {intent_plan.title_suggestion}")
    print(f"{'=' * 60}")

    if cleanup:
        import shutil
        shutil.rmtree(video_work_dir, ignore_errors=True)
        print(f"Cleaned up: {video_work_dir}")

    return output_path


# ---------------------------------------------------------------------------
# Batch mode
# ---------------------------------------------------------------------------

def intent_batch(input_dir, output_dir, **kwargs):
    """Process all videos in a directory with the same intent."""
    input_dir = Path(input_dir)
    if output_dir is None:
        output_dir = str(input_dir) + "_intent"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}
    videos = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in video_exts])

    if not videos:
        print(f"No videos found in {input_dir}")
        return

    print(f"Found {len(videos)} videos to process.")
    results = {"success": [], "failed": []}

    for i, video in enumerate(videos):
        print(f"\n[{i + 1}/{len(videos)}] Processing: {video.name}")
        output_path = str(output_dir / f"{video.stem}_intent.mp4")
        try:
            create_intent_video(str(video), output_path=output_path, **kwargs)
            results["success"].append(video.name)
        except Exception as e:
            print(f"  ERROR: {e}")
            results["failed"].append({"file": video.name, "error": str(e)})

    print(f"\n{'=' * 60}")
    print(f"Batch complete: {len(results['success'])} success, {len(results['failed'])} failed")
    for name in results["success"]:
        print(f"  OK: {name}")
    for item in results["failed"]:
        print(f"  FAIL: {item['file']} -- {item['error']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Intent-driven video content creation with AI analysis and narration"
    )
    parser.add_argument("input", help="Input video file or directory (with --batch)")
    parser.add_argument("--intent", required=True, help="Natural language intent")
    parser.add_argument("--ref-audio", required=True, help="Reference audio for TTS")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--orientation", choices=["portrait", "landscape"], default="portrait")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--work-dir", default="intent_workspace")
    parser.add_argument("--model-dir", default="checkpoints")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--target-duration", type=int, default=180)
    parser.add_argument("--llm-api-key", default=None, help="LLM API key (or LLM_API_KEY env)")
    parser.add_argument("--llm-api-base", default="https://api.openai.com/v1")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--vl-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--cleanup", action="store_true")

    args = parser.parse_args()

    llm_api_key = args.llm_api_key or os.environ.get("LLM_API_KEY")
    use_fp16 = args.fp16

    if not llm_api_key:
        print("Error: LLM API key required. Use --llm-api-key or set LLM_API_KEY env var.")
        sys.exit(1)

    common_kwargs = dict(
        ref_audio=args.ref_audio,
        intent=args.intent,
        orientation=args.orientation,
        work_dir=args.work_dir,
        llm_api_key=llm_api_key,
        llm_api_base=args.llm_api_base,
        llm_model=args.llm_model,
        vl_model=args.vl_model,
        model_dir=args.model_dir,
        target_duration=args.target_duration,
        use_fp16=use_fp16,
        cleanup=args.cleanup,
    )

    if args.batch:
        intent_batch(args.input, args.output, **common_kwargs)
    else:
        create_intent_video(args.input, output_path=args.output, **common_kwargs)


if __name__ == "__main__":
    main()
