import pandas as pd

from enterprise_alert import parse_year_from_period
from indirect_alert import filter_movements_by_quarter, normalize_id_series, str_bool
from main import parse_steps
from utils import normalize_farm_id, parse_quarter_from_period, parse_year_periods


def test_normalize_farm_id_removes_prefix_and_padding():
    assert normalize_farm_id("FARM_ID_00123") == "123"
    assert normalize_farm_id("123.0") == "123"
    assert normalize_farm_id("000456") == "456"
    assert normalize_farm_id(None) == ""


def test_parse_year_periods_keeps_only_valid_ranges():
    assert parse_year_periods("2017-2020, 2021-2023") == ["2017-2020", "2021-2023"]
    assert parse_year_periods("2017, 2018") == []


def test_parse_quarter_from_period_extracts_quarter_from_yyyymm():
    assert parse_quarter_from_period("202401") == 1
    assert parse_quarter_from_period("202404") == 4
    assert parse_quarter_from_period("2024") is None


def test_normalize_id_series_applies_normalization_rules():
    sr = pd.Series(["FARM_ID_00123", "00456", "123.0", "ABC"])
    result = normalize_id_series(sr, {"strip_leading_zeros": True, "strip_dot_zero": True})
    assert result.tolist() == ["123", "456", "123", "ABC"]


def test_str_bool_handles_common_truthy_and_falsey_values():
    assert str_bool("true") is True
    assert str_bool("yes") is True
    assert str_bool("0") is False
    assert str_bool("no") is False
    assert str_bool("nan") is False


def test_filter_movements_by_quarter_keeps_only_requested_quarter():
    df = pd.DataFrame(
        {
            "DATE": ["2024-01-10", "2024-04-15", "2024-07-20", "2024-10-01"],
            "id": [1, 2, 3, 4],
        }
    )

    result = filter_movements_by_quarter(df, quarter=2)

    assert list(result["id"]) == [2]
    assert "quarter" not in result.columns


def test_parse_year_from_period_handles_supported_formats():
    assert parse_year_from_period("2010-2024", "cumulative") == 2024
    assert parse_year_from_period("201701", "nad") == 2017
    assert parse_year_from_period("2017", "annual") == 2017


def test_parse_steps_supports_ranges_and_all():
    assert parse_steps(["1-3", "5"]) == [1, 2, 3, 5]
    assert parse_steps(["all"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert parse_steps(["0", "bad", "7-9"]) == [7, 8, 9]
