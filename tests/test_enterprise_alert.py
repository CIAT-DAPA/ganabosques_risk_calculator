import pandas as pd

import enterprise_alert


def test_parse_year_from_period():
    assert enterprise_alert.parse_year_from_period("2010-2024", "cumulative") == 2024
    assert enterprise_alert.parse_year_from_period("201701", "nad") == 2017
    assert enterprise_alert.parse_year_from_period("2017", "annual") == 2017


def test_calculate_enterprise_alerts_for_period_writes_csv(tmp_path, monkeypatch):
    direct_dir = tmp_path / "direct"
    movement_dir = tmp_path / "movement"
    output_dir = tmp_path / "out"
    direct_dir.mkdir(parents=True)
    movement_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)

    pd.DataFrame({"id": ["00123"], "direct_alert": [True]}).to_csv(direct_dir / "smbyc_direct_alert_nad_202401.csv", index=False)
    pd.DataFrame({
        "SIT_CODE_ORIGEN": ["123"],
        "SIT_CODE_DESTINO": ["999"],
        "TIPO_ORIGEN": ["FARM"],
        "TIPO_DESTINO": ["SLAUGHTERHOUSE"],
        "PRODUCER_ID_ORIGEN": ["p1"],
        "PRODUCER_ID_DESTINO": ["p2"],
        "DATE": ["2024-01-10"],
    }).to_csv(movement_dir / "movement_data_base_2024.csv", index=False)

    def fake_pkg_alert_enterprise(total_risk_df, movements_df, id_column, normalize_ids=True, show_progress=True):
        return pd.DataFrame({
            "idpro": ["E1"],
            "id_farm": ["123"],
            "typemove": ["in"],
            "enterprise_type": ["SLAUGHTERHOUSE"],
        })

    monkeypatch.setattr(enterprise_alert, "pkg_alert_enterprise", fake_pkg_alert_enterprise)
    monkeypatch.setattr(enterprise_alert, "load_enterprise_mapping", lambda *args, **kwargs: {})

    result = enterprise_alert.calculate_enterprise_alerts_for_period(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_dir=str(direct_dir),
        movement_csv_dir=str(movement_dir),
        output_dir=str(output_dir),
    )

    assert result["success"] is True
    output_file = output_dir / "smbyc_enterprise_alert_nad_202401.csv"
    assert output_file.exists()
