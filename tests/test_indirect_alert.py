import pandas as pd

import indirect_alert


def test_normalize_id_series_and_str_bool():
    sr = pd.Series(["FARM_ID_00123", "00456", "123.0", "ABC"])
    result = indirect_alert.normalize_id_series(sr, {"strip_leading_zeros": True, "strip_dot_zero": True})
    assert result.tolist() == ["123", "456", "123", "ABC"]
    assert indirect_alert.str_bool("true") is True
    assert indirect_alert.str_bool("no") is False


def test_filter_movements_by_quarter_and_load_direct_alerts(tmp_path):
    df = pd.DataFrame(
        {
            "DATE": ["2024-01-10", "2024-04-15", "2024-07-20", "2024-10-01"],
            "id": [1, 2, 3, 4],
        }
    )
    result = indirect_alert.filter_movements_by_quarter(df, quarter=2)
    assert list(result["id"]) == [2]

    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["00123", "00456"], "direct_alert": ["true", "false"]}).to_csv(direct_csv, index=False)
    loaded = indirect_alert.load_direct_alerts(str(direct_csv), {"strip_leading_zeros": True, "strip_dot_zero": True})
    assert loaded["id"].tolist() == ["123", "456"]
    assert loaded["direct_alert"].tolist() == [True, False]


def test_extract_enterprise_alerts():
    movements = pd.DataFrame(
        {
            "origen_id": ["123", "456"],
            "destination_id": ["999", "123"],
            "tipo_origen": ["FARM", "SLAUGHTERHOUSE"],
            "tipo_destino": ["SLAUGHTERHOUSE", "FARM"],
            "producer_id_origen": ["p1", "p2"],
            "producer_id_destino": ["p3", "p4"],
        }
    )
    enterprise_df = indirect_alert.extract_enterprise_alerts(
        movements,
        farms_with_alert={"123"},
        period="202401",
        year="2024",
        quarter=1,
    )
    assert not enterprise_df.empty
    assert set(enterprise_df["typemove"]) == {"in", "out"}
