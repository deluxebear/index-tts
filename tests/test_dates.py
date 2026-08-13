from dub_pipeline import normalize_spoken_dates


def test_month_day_year():
    assert normalize_spoken_dates("We met on January 15, 2024.") == "We met on 2024年1月15日."
    assert normalize_spoken_dates("Jan. 3rd, 2020") == "2020年1月3日"
    assert normalize_spoken_dates("15 January 2024") == "2024年1月15日"


def test_partial_dates():
    assert normalize_spoken_dates("the 3rd of March") == "3月3日"
    assert normalize_spoken_dates("March 3rd") == "3月3日"
    assert normalize_spoken_dates("March 2020") == "2020年3月"


def test_numeric_and_iso():
    assert normalize_spoken_dates("due 2024-01-15") == "due 2024年1月15日"
    assert normalize_spoken_dates("15/03/2024") == "2024年3月15日"
    assert normalize_spoken_dates("03/15/2024") == "2024年3月15日"


def test_idempotent_and_non_dates():
    assert normalize_spoken_dates("2024年1月15日") == "2024年1月15日"
    assert normalize_spoken_dates("May I continue?") == "May I continue?"
    assert normalize_spoken_dates("") == ""
