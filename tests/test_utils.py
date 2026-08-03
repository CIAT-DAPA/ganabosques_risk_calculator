import pandas as pd

from utils import normalize_farm_id, parse_quarter_from_period, parse_year_periods


def test_normalize_farm_id_removes_prefix_and_padding():
    assert normalize_farm_id("FARM_ID_00123") == "123"
    assert normalize_farm_id("123.0") == "123"
    assert normalize_farm_id("000456") == "456"
    assert normalize_farm_id(None) == ""


def test_parse_year_periods_and_quarter():
    assert parse_year_periods("2017-2020, 2021-2023") == ["2017-2020", "2021-2023"]
    assert parse_year_periods("2017, 2018") == []
    assert parse_quarter_from_period("202401") == 1
    assert parse_quarter_from_period("202404") == 4
    assert parse_quarter_from_period("2024") is None
