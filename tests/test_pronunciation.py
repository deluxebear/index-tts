from dub_pipeline import (
    annotate_tts_text,
    find_suspicious_segments,
    load_pronunciation_glossary,
    list_suspicious_segments,
)


def test_glossary_and_existing_tags(tmp_path):
    yaml_path = tmp_path / "pronunciation.yaml"
    yaml_path.write_text(
        "银行: 银<行|HANG2>\nChatGPT: <ChatGPT|CH AE1 T JH IY1 P IY1 T IY1>\n",
        encoding="utf-8",
    )
    glossary = load_pronunciation_glossary(path=str(yaml_path))
    assert annotate_tts_text("他在银行办理业务", glossary=glossary, use_g2p=False) == (
        "他在银<行|HANG2>办理业务"
    )
    assert annotate_tts_text("银<行|HANG2>", glossary=glossary, use_g2p=False) == (
        "银<行|HANG2>"
    )
    assert annotate_tts_text("我们用ChatGPT", glossary=glossary, use_g2p=False) == (
        "我们用<ChatGPT|CH AE1 T JH IY1 P IY1 T IY1>"
    )


def test_g2p_wraps_leftover_english(monkeypatch):
    class _FakeG2p:
        def __call__(self, word):
            return ["T", "EH1", "S", "L", "AH0"]

    monkeypatch.setattr("dub_pipeline._G2P", _FakeG2p())
    monkeypatch.setattr("dub_pipeline._G2P_UNAVAILABLE", False)
    out = annotate_tts_text("我们用 Tesla", glossary={}, use_g2p=True)
    assert out == "我们用 <Tesla|T EH1 S L AH0>"


def test_find_suspicious_segments():
    segs = [
        {"start": 0.0, "end": 1.0, "text": "Hello", "zh_text": "Hello"},
        {"start": 1.0, "end": 2.0, "text": "Hi", "zh_text": "你好"},
        {"start": 2.0, "end": 3.0, "text": "Bye", "zh_text": ""},
        {
            "start": 3.0, "end": 4.0, "text": "x",
            "zh_text": "这是一句非常非常非常非常非常非常非常非常长的话用来检查语速过快",
        },
        {"start": 5.0, "end": 6.0, "text": "Mon", "zh_text": "See you Monday at 3:00 PM for $100"},
    ]
    rows = find_suspicious_segments(segs)
    ids = {r["id"] for r in rows}
    assert 0 in ids
    assert 1 not in ids
    assert 2 in ids
    assert 3 in ids
    assert 4 in ids
    reasons = {r["id"]: set(r["reasons"]) for r in rows}
    assert "leftover_english" in reasons[0]
    assert "empty_zh" in reasons[2]
    assert "too_fast" in reasons[3]
    assert "leftover_weekday" in reasons[4] or "leftover_datetime" in reasons[4]


def test_list_suspicious_from_checkpoint(tmp_path, capsys):
    import json

    video = tmp_path / "talk.mp4"
    video.write_bytes(b"x")
    vwd = tmp_path / "ws" / "talk"
    vwd.mkdir(parents=True)
    (vwd / "checkpoint.json").write_text(json.dumps({
        "step": 10,
        "segments": [
            {"start": 0.0, "end": 1.0, "text": "Hello", "zh_text": "Hello", "speaker": "A"},
        ],
        "paths": {},
    }), encoding="utf-8")
    rows = list_suspicious_segments(str(video), work_dir=str(tmp_path / "ws"))
    assert len(rows) == 1
    assert "leftover_english" in rows[0]["reasons"]
    assert "[  0]" in capsys.readouterr().out
