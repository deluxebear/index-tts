from dub_pipeline import normalize_spoken_datetime, normalize_spoken_times


def test_ampm_clock():
    assert normalize_spoken_times("at 3:00 PM") == "at 下午3点"
    assert normalize_spoken_times("3:30 p.m.") == "下午3点半"
    assert normalize_spoken_times("10:15 AM") == "上午10点15分"
    assert normalize_spoken_times("3pm") == "下午3点"
    assert normalize_spoken_times("12:00 PM") == "中午12点"
    assert normalize_spoken_times("12:00 AM") == "凌晨12点"


def test_24h_and_phrases():
    assert normalize_spoken_times("starts 15:00") == "starts 下午3点"
    assert normalize_spoken_times("3 o'clock") == "3点"
    assert normalize_spoken_times("half past 3") == "3点半"
    assert normalize_spoken_times("quarter past three") == "3点15分"
    assert normalize_spoken_times("quarter to 5") == "4点45分"


def test_skip_ratios_and_already_chinese():
    assert normalize_spoken_times("ratio 3:1") == "ratio 3:1"
    assert normalize_spoken_times("下午3点") == "下午3点"
    assert normalize_spoken_times("") == ""


def test_date_then_time():
    assert (
        normalize_spoken_datetime("January 15, 2024 at 3:00 PM")
        == "2024年1月15日 at 下午3点"
    )
