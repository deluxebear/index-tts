from dub_pipeline import translate_with_context, untranslated_segment_ids


class _EmptyLLM:
    def chat(self, prompt):
        return "sorry I cannot"


class _PartialLLM:
    def chat(self, prompt):
        return "#0 你好"


def test_translation_does_not_fall_back_to_english():
    segs = [
        {"text": "Hello world", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"text": "Thanks", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
    ]
    out = translate_with_context(segs, {"topic": "t"}, _EmptyLLM(), batch_size=12)
    assert out[0].get("zh_text") in (None, "")
    assert out[1].get("zh_text") in (None, "")
    assert out[0].get("text") == "Hello world"
    assert untranslated_segment_ids(out) == [0, 1]


def test_translation_keeps_successes_and_lists_failures():
    segs = [
        {"text": "Hello world", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"text": "Thanks", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
    ]
    out = translate_with_context(segs, {"topic": "t"}, _PartialLLM(), batch_size=12)
    assert out[0]["zh_text"] == "你好"
    assert out[1].get("zh_text") in (None, "")
    assert untranslated_segment_ids(out) == [1]


class _EnglishLLM:
    def chat(self, prompt):
        return "#0 Hello everyone\n#1 Thanks a lot"


def test_english_looking_translation_is_rejected():
    segs = [
        {"text": "Hello world", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"text": "Thanks", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_00"},
    ]
    out = translate_with_context(segs, {"topic": "t"}, _EnglishLLM(), batch_size=12)
    assert out[0].get("zh_text") in (None, "")
    assert out[1].get("zh_text") in (None, "")
    assert untranslated_segment_ids(out) == [0, 1]
