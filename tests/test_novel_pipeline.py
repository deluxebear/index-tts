import json
import re

import wave

from novel_pipeline import (
    assign_seed_voices,
    assign_silence,
    build_card_text,
    build_chapter_script,
    enforce_tts_limits,
    extract_characters,
    generate_character_voices,
    ingest_text,
    merge_chapter,
    merge_character_lists,
    prepare_tts_text,
    split_chapters,
    split_utterances,
    synthesize_chapter,
    validate_character,
    validate_style,
)



def test_split_chapters_chinese_heading():
    text = open("tests/fixtures/novel_sample.txt", encoding="utf-8").read()
    chapters = split_chapters(ingest_text(text), "novel_sample.txt")
    assert [c["id"] for c in chapters] == ["c01", "c02"]
    assert chapters[0]["title"].startswith("第一章")
    assert "张三站在巷口" in text[chapters[0]["start_char"]:chapters[0]["end_char"]]
    assert "雨停了" in text[chapters[1]["start_char"]:chapters[1]["end_char"]]


def test_ingest_strips_bom_and_crlf():
    assert ingest_text("\ufeffhello\r\nworld\r\n") == "hello\nworld\n"


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
    assert all("<行|HANG2>" in p or "<行|HANG2>" not in text for p in parts) or any(
        "<行|HANG2>" in p for p in parts
    )
    assert all("<行|" not in p or "|HANG2>" in p for p in parts)


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


def test_extract_characters_sends_full_midsize_chapter_body():
    """Chapters of 4001–6000 chars must not be re-truncated to 4000 in the LLM prompt."""
    marker = "TAIL_MARKER_XYZ"
    body = ("甲" * (5000 - len(marker))) + marker
    assert len(body) == 5000
    chapters = [{"id": "c01", "start_char": 0, "end_char": len(body)}]
    seen: list[str] = []

    class FakeLLM:
        def chat(self, prompt: str) -> str:
            seen.append(prompt)
            return "[]"

    extract_characters(body, chapters, FakeLLM())
    assert len(seen) == 1
    assert marker in seen[0]
    assert body in seen[0]


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


def test_build_chapter_script_resplits_oversized_llm_tts_text(tmp_path):
    """LLM-rewritten tts_text longer than 80 must be re-split with speaker/emo kept."""
    long_tts = "甲" * 150
    emo = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    class FakeLLM:
        def chat(self, prompt: str) -> str:
            return json.dumps([{
                "index": 0,
                "speaker_id": "narrator",
                "emo_vector": emo,
                "tts_text": long_tts,
            }], ensure_ascii=False)

    characters = [{
        "id": "narrator", "name": "旁白", "role": "narrator",
        "gender": "male", "age": "middle", "personality": "沉", "voice_traits": "低",
    }]
    out = build_chapter_script(
        "短句。", "c01", characters, {"lang": "zh"}, FakeLLM(), str(tmp_path),
    )
    assert len(out) >= 2
    pron_re = re.compile(r"<[^|>]+\|[^>]+>")
    for u in out:
        visible = pron_re.sub("X", u["tts_text"] or "")
        assert len(visible) <= 80
        assert u["speaker_id"] == "narrator"
        assert u["emo_vector"] == emo


def _write_wav(path, nframes=2205):
    with wave.open(str(path), "w") as w:
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
        "emo_vector": [0] * 8, "duration_factor": 1.0, "silence_after_ms": 200,
    }]
    synthesize_chapter(utts, {"zhang_san": {"ref_wav": "ref.wav"}}, BoomTTS(), str(tmp_path))


def test_merge_chapter_inserts_silence(tmp_path):
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    out = tmp_path / "c.wav"
    _write_wav(a, 2205)
    _write_wav(b, 2205)
    merge_chapter(
        [{"wav_path": str(a), "silence_after_ms": 1000},
         {"wav_path": str(b), "silence_after_ms": 0}],
        str(out),
    )
    with wave.open(str(out)) as w:
        assert w.getnframes() == 2205 + 22050 + 2205


