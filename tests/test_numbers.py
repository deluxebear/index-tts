from dub_pipeline import (
    CHARS_PER_SECOND,
    normalize_spoken_datetime,
    normalize_spoken_numbers,
    normalize_spoken_weekdays,
)


def test_chars_per_second_is_slower_spoken_rate():
    assert CHARS_PER_SECOND == 3.8


def test_money_and_percent():
    assert normalize_spoken_numbers("costs $100") == "costs 100美元"
    assert normalize_spoken_numbers("USD 100") == "100美元"
    assert normalize_spoken_numbers("100 dollars") == "100美元"
    assert normalize_spoken_numbers("€50") == "50欧元"
    assert normalize_spoken_numbers("50 euros") == "50欧元"
    assert normalize_spoken_numbers("£20") == "20英镑"
    assert normalize_spoken_numbers("20 pounds") == "20英镑"
    assert normalize_spoken_numbers("¥100") == "100元"
    assert normalize_spoken_numbers("100 yuan") == "100元"
    assert normalize_spoken_numbers("50%") == "百分之50"
    assert normalize_spoken_numbers("50 percent") == "百分之50"
    assert normalize_spoken_numbers("$1,000") == "1000美元"


def test_million_billion():
    assert normalize_spoken_numbers("$1.5 million") == "150万美元"
    assert normalize_spoken_numbers("$1.5m") == "150万美元"
    assert normalize_spoken_numbers("1.5 million dollars") == "150万美元"
    assert normalize_spoken_numbers("1 million") == "100万"
    assert normalize_spoken_numbers("2 billion") == "20亿"
    assert normalize_spoken_numbers("1 billion dollars") == "10亿美元"


def test_weekdays():
    assert normalize_spoken_weekdays("on Monday") == "on 星期一"
    assert normalize_spoken_weekdays("Tue.") == "星期二"
    assert normalize_spoken_weekdays("Sunday") == "星期日"
    assert normalize_spoken_weekdays("this weekend") == "this 周末"


def test_numbers_idempotent_and_non_targets():
    assert normalize_spoken_numbers("100美元") == "100美元"
    assert normalize_spoken_numbers("百分之50") == "百分之50"
    assert normalize_spoken_numbers("") == ""
    assert normalize_spoken_weekdays("Sunshine") == "Sunshine"
    assert normalize_spoken_weekdays("星期一") == "星期一"


def test_datetime_chains_numbers_and_weekdays():
    assert (
        normalize_spoken_datetime("Monday, January 15, 2024 at 3:00 PM, $1.5 million")
        == "星期一, 2024年1月15日 at 下午3点, 150万美元"
    )
