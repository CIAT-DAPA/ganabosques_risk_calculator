import pandas as pd
import pytest

import indirect_alert


NORMALIZE = {"strip_leading_zeros": True, "strip_dot_zero": True}


# ===================== utilidades =====================


def test_normalize_id_series_applies_configured_rules():
    sr = pd.Series(["FARM_ID_00123", "00456", "123.0", "ABC"])

    assert indirect_alert.normalize_id_series(sr, NORMALIZE).tolist() == [
        "123",
        "456",
        "123",
        "ABC",
    ]

    sin_ceros = indirect_alert.normalize_id_series(
        sr, {"strip_leading_zeros": False, "strip_dot_zero": False}
    )
    assert sin_ceros.tolist() == ["00123", "00456", "123.0", "ABC"]

    # Sin claves explícitas se asumen ambas activas.
    assert indirect_alert.normalize_id_series(pd.Series(["00789"]), {}).tolist() == ["789"]


@pytest.mark.parametrize(
    "value", ["true", "True", "T", "1", "yes", "y", "si", "sí", " TRUE "]
)
def test_str_bool_truthy_values(value):
    assert indirect_alert.str_bool(value) is True


@pytest.mark.parametrize("value", ["false", "F", "0", "no", "n", "", "nan"])
def test_str_bool_falsey_values(value):
    assert indirect_alert.str_bool(value) is False


def test_str_bool_falls_back_to_truthiness_for_unknown_strings():
    assert indirect_alert.str_bool("cualquier cosa") is True
    # OJO: None se convierte a la cadena "none", que no está en ninguna de las dos
    # listas y por tanto se evalúa como truthy. Es contraintuitivo, pero es el
    # comportamiento actual; se fija aquí para que un cambio no pase inadvertido.
    assert indirect_alert.str_bool(None) is True


def test_filter_movements_by_quarter_keeps_only_requested_quarter():
    df = pd.DataFrame(
        {
            "DATE": ["2024-01-10", "2024-04-15", "2024-07-20", "2024-10-01"],
            "id": [1, 2, 3, 4],
        }
    )

    result = indirect_alert.filter_movements_by_quarter(df, quarter=2)

    assert result["id"].tolist() == [2]
    # La columna auxiliar `quarter` no se filtra al resultado.
    assert "quarter" not in result.columns


def test_filter_movements_by_quarter_uses_custom_date_column():
    df = pd.DataFrame({"fecha": ["2024-01-10", "2024-07-20"], "id": [1, 2]})

    result = indirect_alert.filter_movements_by_quarter(df, quarter=3, date_column="fecha")

    assert result["id"].tolist() == [2]


def test_filter_movements_by_quarter_returns_input_on_error(monkeypatch):
    df = pd.DataFrame({"DATE": ["2024-01-10"], "id": [1]})

    monkeypatch.setattr(
        indirect_alert.pd,
        "to_datetime",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("fecha ilegible")),
    )

    result = indirect_alert.filter_movements_by_quarter(df, quarter=1)

    assert result.equals(df)


# ===================== load_direct_alerts =====================


def test_load_direct_alerts_returns_empty_frame_for_missing_file(tmp_path):
    df = indirect_alert.load_direct_alerts(str(tmp_path / "no.csv"), NORMALIZE)

    assert df.empty
    assert list(df.columns) == ["id", "direct_alert"]


def test_load_direct_alerts_returns_empty_frame_without_id_column(tmp_path):
    csv_path = tmp_path / "direct.csv"
    pd.DataFrame({"otra": ["x"], "direct_alert": ["true"]}).to_csv(csv_path, index=False)

    assert indirect_alert.load_direct_alerts(str(csv_path), NORMALIZE).empty


def test_load_direct_alerts_returns_empty_frame_without_alert_column(tmp_path):
    csv_path = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["1"], "deforested_ha": ["2.5"]}).to_csv(csv_path, index=False)

    assert indirect_alert.load_direct_alerts(str(csv_path), NORMALIZE).empty


@pytest.mark.parametrize(
    "column",
    ["direct_alert", "intersect_deforestation", "intersect_early_warnings"],
)
def test_load_direct_alerts_accepts_each_supported_alert_column(tmp_path, column):
    csv_path = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["00123", "00456"], column: ["true", "false"]}).to_csv(
        csv_path, index=False
    )

    df = indirect_alert.load_direct_alerts(str(csv_path), NORMALIZE)

    assert df["id"].tolist() == ["123", "456"]
    assert df["direct_alert"].tolist() == [True, False]
    assert list(df.columns) == ["id", "direct_alert"]


def test_load_direct_alerts_returns_empty_frame_on_read_error(tmp_path, monkeypatch):
    csv_path = tmp_path / "direct.csv"
    csv_path.write_text("id,direct_alert\n1,true\n", encoding="utf-8")
    monkeypatch.setattr(
        indirect_alert.pd,
        "read_csv",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("csv corrupto")),
    )

    assert indirect_alert.load_direct_alerts(str(csv_path), NORMALIZE).empty


# ===================== load_movements =====================


def _write_movements(path, with_enterprise=True):
    rows = {
        "SIT_CODE_ORIGEN": ["FARM_ID_00123", "999", "00777"],
        "SIT_CODE_DESTINO": ["999", "00456", "888"],
        "DATE": ["2024-01-10", "2024-05-20", "2024-02-02"],
    }
    if with_enterprise:
        rows.update(
            {
                "TIPO_ORIGEN": ["FARM", "SLAUGHTERHOUSE", "FARM"],
                "TIPO_DESTINO": ["SLAUGHTERHOUSE", "FARM", "CATTLE_FAIR"],
                "PRODUCER_ID_ORIGEN": ["00001", "E2", "00003"],
                "PRODUCER_ID_DESTINO": ["E2", "00001", "E4"],
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_movements_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No existe archivo de movimientos"):
        indirect_alert.load_movements(str(tmp_path / "no.csv"), NORMALIZE)


def test_load_movements_raises_when_required_columns_are_missing(tmp_path):
    csv_path = tmp_path / "movement.csv"
    pd.DataFrame({"OTRA": ["x"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="SIT_CODE_ORIGEN"):
        indirect_alert.load_movements(str(csv_path), NORMALIZE)


def test_load_movements_returns_only_id_columns_by_default(tmp_path):
    csv_path = _write_movements(tmp_path / "movement.csv")

    df = indirect_alert.load_movements(str(csv_path), NORMALIZE)

    assert list(df.columns) == ["origen_id", "destination_id"]
    assert df["origen_id"].tolist() == ["123", "999", "777"]
    assert df["destination_id"].tolist() == ["999", "456", "888"]


def test_load_movements_includes_enterprise_fields_when_requested(tmp_path):
    csv_path = _write_movements(tmp_path / "movement.csv")

    df = indirect_alert.load_movements(str(csv_path), NORMALIZE, include_enterprise_fields=True)

    assert list(df.columns) == [
        "origen_id",
        "destination_id",
        "tipo_origen",
        "tipo_destino",
        "producer_id_origen",
        "producer_id_destino",
    ]
    # Los IDs de productor también se normalizan.
    assert df["producer_id_origen"].tolist() == ["1", "E2", "3"]
    assert df["producer_id_destino"].tolist() == ["E2", "1", "E4"]


def test_load_movements_skips_absent_enterprise_columns(tmp_path):
    csv_path = _write_movements(tmp_path / "movement.csv", with_enterprise=False)

    df = indirect_alert.load_movements(str(csv_path), NORMALIZE, include_enterprise_fields=True)

    assert list(df.columns) == ["origen_id", "destination_id"]


def test_load_movements_filters_by_quarter(tmp_path):
    csv_path = _write_movements(tmp_path / "movement.csv")

    df = indirect_alert.load_movements(str(csv_path), NORMALIZE, quarter=1)

    assert df["origen_id"].tolist() == ["123", "777"]


# ===================== extract_enterprise_alerts =====================


def _enterprise_movements():
    return pd.DataFrame(
        {
            "origen_id": ["123", "456", "999"],
            "destination_id": ["E1", "123", "888"],
            "tipo_origen": ["FARM", "SLAUGHTERHOUSE", "FARM"],
            "tipo_destino": ["SLAUGHTERHOUSE", "FARM", "FARM"],
            "producer_id_origen": ["p1", "p2", "p3"],
            "producer_id_destino": ["p4", "p5", "p6"],
        }
    )


def test_extract_enterprise_alerts_detects_entries_and_exits():
    df = indirect_alert.extract_enterprise_alerts(
        _enterprise_movements(),
        farms_with_alert={"123"},
        period="202401",
        year="2024",
        quarter=1,
    )

    assert set(df["typemove"]) == {"in", "out"}
    entrada = df[df["typemove"] == "in"].iloc[0]
    salida = df[df["typemove"] == "out"].iloc[0]

    # Entrada: finca alertada → empresa. El idpro es el productor de destino.
    assert entrada["id_farm"] == "123"
    assert entrada["idpro"] == "p4"
    assert entrada["type_enterprise"] == "SLAUGHTERHOUSE"
    # Salida: empresa → finca alertada. El idpro es el productor de origen.
    assert salida["id_farm"] == "123"
    assert salida["idpro"] == "p2"
    assert salida["type_enterprise"] == "SLAUGHTERHOUSE"

    assert df["period"].tolist() == ["202401", "202401"]
    assert df["year"].tolist() == ["2024", "2024"]
    assert df["quarter"].tolist() == [1, 1]
    assert df["farm_has_direct_alert"].all()


def test_extract_enterprise_alerts_omits_quarter_when_not_given():
    df = indirect_alert.extract_enterprise_alerts(
        _enterprise_movements(), farms_with_alert={"123"}, period="2024", year="2024"
    )

    assert "quarter" not in df.columns


def test_extract_enterprise_alerts_ignores_farm_to_farm_movements():
    movements = pd.DataFrame(
        {
            "origen_id": ["123"],
            "destination_id": ["456"],
            "tipo_origen": ["FARM"],
            "tipo_destino": ["FARM"],
            "producer_id_origen": ["p1"],
            "producer_id_destino": ["p2"],
        }
    )

    df = indirect_alert.extract_enterprise_alerts(
        movements, farms_with_alert={"123", "456"}, period="2024", year="2024"
    )

    assert df.empty


def test_extract_enterprise_alerts_deduplicates_identical_rows():
    movements = pd.concat([_enterprise_movements()] * 3, ignore_index=True)

    df = indirect_alert.extract_enterprise_alerts(
        movements, farms_with_alert={"123"}, period="202401", year="2024", quarter=1
    )

    assert len(df) == 2


def test_extract_enterprise_alerts_returns_empty_without_type_columns():
    movements = pd.DataFrame({"origen_id": ["123"], "destination_id": ["E1"]})

    df = indirect_alert.extract_enterprise_alerts(
        movements, farms_with_alert={"123"}, period="2024", year="2024"
    )

    assert df.empty


def test_extract_enterprise_alerts_needs_producer_columns_too():
    movements = pd.DataFrame(
        {
            "origen_id": ["123"],
            "destination_id": ["E1"],
            "tipo_origen": ["FARM"],
            "tipo_destino": ["SLAUGHTERHOUSE"],
        }
    )

    # Hay tipo pero no productor: no se puede identificar la empresa.
    df = indirect_alert.extract_enterprise_alerts(
        movements, farms_with_alert={"123"}, period="2024", year="2024"
    )

    assert df.empty


# ===================== calculate_indirect_alerts =====================


@pytest.fixture
def indirect_dirs(tmp_path):
    movement = tmp_path / "movements"
    out = tmp_path / "results" / "indirect_alerts"
    movement.mkdir(parents=True)
    out.mkdir(parents=True)
    return tmp_path, movement, out


def _fake_pkg_indirect(**kwargs):
    return pd.DataFrame(
        {
            "id": ["123", "456"],
            "n_total_mov": [2, 1],
            "n_in": [1, 1],
            "n_out": [1, 0],
            "n_indirect_in": [1, 0],
            "n_indirect_out": [0, 0],
            "indirect_alert_in": [True, False],
            "indirect_alert_out": [False, False],
        }
    )


def test_calculate_indirect_alerts_writes_both_outputs(indirect_dirs, monkeypatch):
    tmp_path, movement_dir, out_dir = indirect_dirs
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "direct_alert": ["true"]}).to_csv(direct_csv, index=False)
    _write_movements(movement_dir / "movement_data_base_2024.csv")

    monkeypatch.setattr(indirect_alert, "pkg_alert_indirect", _fake_pkg_indirect)

    output_csv = out_dir / "smbyc_indirect_alert_nad_202401.csv"
    result = indirect_alert.calculate_indirect_alerts(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_csv=str(direct_csv),
        movement_csv_dir=str(movement_dir),
        output_csv=str(output_csv),
    )

    assert result["success"] is True
    assert result["year"] == "2024"
    assert result["quarter"] == 1
    assert result["farms_with_direct_alert"] == 1
    assert result["indirect_alert_in_count"] == 1
    assert result["indirect_alert_out_count"] == 0
    assert result["enterprise_generated"] is True

    written = pd.read_csv(output_csv, dtype={"period": str, "year": str})
    # La metadata se inserta al principio, en orden.
    assert list(written.columns)[:3] == ["period", "year", "quarter"]

    ent_path = out_dir.parent / "enterprise_alerts" / "smbyc_enterprise_alert_nad_202401.csv"
    assert ent_path.exists()


def test_calculate_indirect_alerts_uses_explicit_enterprise_path(indirect_dirs, monkeypatch):
    tmp_path, movement_dir, out_dir = indirect_dirs
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "direct_alert": ["true"]}).to_csv(direct_csv, index=False)
    _write_movements(movement_dir / "movement_data_base_2024.csv")
    monkeypatch.setattr(indirect_alert, "pkg_alert_indirect", _fake_pkg_indirect)

    ent_csv = tmp_path / "custom" / "empresas.csv"

    result = indirect_alert.calculate_indirect_alerts(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_csv=str(direct_csv),
        movement_csv_dir=str(movement_dir),
        output_csv=str(out_dir / "out.csv"),
        enterprise_output_csv=str(ent_csv),
    )

    assert result["enterprise_output_file"] == str(ent_csv)
    assert ent_csv.exists()


def test_calculate_indirect_alerts_can_skip_enterprise_generation(indirect_dirs, monkeypatch):
    tmp_path, movement_dir, out_dir = indirect_dirs
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "direct_alert": ["true"]}).to_csv(direct_csv, index=False)
    _write_movements(movement_dir / "movement_data_base_2024.csv")
    monkeypatch.setattr(indirect_alert, "pkg_alert_indirect", _fake_pkg_indirect)

    result = indirect_alert.calculate_indirect_alerts(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_csv=str(direct_csv),
        movement_csv_dir=str(movement_dir),
        output_csv=str(out_dir / "out.csv"),
        generate_enterprise_alerts=False,
    )

    assert result["enterprise_generated"] is False
    assert not (out_dir.parent / "enterprise_alerts").exists()


def test_calculate_indirect_alerts_reports_when_no_enterprise_movements(indirect_dirs, monkeypatch):
    tmp_path, movement_dir, out_dir = indirect_dirs
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "direct_alert": ["true"]}).to_csv(direct_csv, index=False)
    # Solo movimientos finca→finca: no hay empresas involucradas.
    pd.DataFrame(
        {
            "SIT_CODE_ORIGEN": ["123"],
            "SIT_CODE_DESTINO": ["456"],
            "DATE": ["2024-01-10"],
            "TIPO_ORIGEN": ["FARM"],
            "TIPO_DESTINO": ["FARM"],
            "PRODUCER_ID_ORIGEN": ["p1"],
            "PRODUCER_ID_DESTINO": ["p2"],
        }
    ).to_csv(movement_dir / "movement_data_base_2024.csv", index=False)
    monkeypatch.setattr(indirect_alert, "pkg_alert_indirect", _fake_pkg_indirect)

    result = indirect_alert.calculate_indirect_alerts(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_csv=str(direct_csv),
        movement_csv_dir=str(movement_dir),
        output_csv=str(out_dir / "out.csv"),
    )

    assert result["success"] is True
    assert result["enterprise_generated"] is False


def test_calculate_indirect_alerts_cumulative_uses_year_before_range_end(indirect_dirs, monkeypatch):
    tmp_path, movement_dir, out_dir = indirect_dirs
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "direct_alert": ["true"]}).to_csv(direct_csv, index=False)
    # Para el acumulado 2010-2024 el pipeline busca los movimientos de 2023.
    _write_movements(movement_dir / "movement_data_base_2023.csv")
    monkeypatch.setattr(indirect_alert, "pkg_alert_indirect", _fake_pkg_indirect)

    result = indirect_alert.calculate_indirect_alerts(
        period="2010-2024",
        period_type="cumulative",
        source="smbyc",
        direct_alerts_csv=str(direct_csv),
        movement_csv_dir=str(movement_dir),
        output_csv=str(out_dir / "out.csv"),
    )

    assert result["success"] is True
    assert result["year"] == 2023
    assert result["quarter"] is None


def test_calculate_indirect_alerts_annual_has_no_quarter(indirect_dirs, monkeypatch):
    tmp_path, movement_dir, out_dir = indirect_dirs
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "direct_alert": ["true"]}).to_csv(direct_csv, index=False)
    _write_movements(movement_dir / "movement_data_base_2024.csv")
    monkeypatch.setattr(indirect_alert, "pkg_alert_indirect", _fake_pkg_indirect)

    result = indirect_alert.calculate_indirect_alerts(
        period="2024",
        period_type="annual",
        source="smbyc",
        direct_alerts_csv=str(direct_csv),
        movement_csv_dir=str(movement_dir),
        output_csv=str(out_dir / "out.csv"),
    )

    assert result["quarter"] is None
    written = pd.read_csv(out_dir / "out.csv")
    assert "quarter" not in written.columns


def test_calculate_indirect_alerts_fails_without_direct_alerts(indirect_dirs):
    tmp_path, movement_dir, out_dir = indirect_dirs

    result = indirect_alert.calculate_indirect_alerts(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_csv=str(tmp_path / "no.csv"),
        movement_csv_dir=str(movement_dir),
        output_csv=str(out_dir / "out.csv"),
    )

    assert result == {
        "success": False,
        "period": "202401",
        "error": "No direct alerts found",
    }


def test_calculate_indirect_alerts_fails_without_movement_file(indirect_dirs):
    tmp_path, movement_dir, out_dir = indirect_dirs
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "direct_alert": ["true"]}).to_csv(direct_csv, index=False)

    result = indirect_alert.calculate_indirect_alerts(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_csv=str(direct_csv),
        movement_csv_dir=str(movement_dir),
        output_csv=str(out_dir / "out.csv"),
    )

    assert result["success"] is False
    assert result["error"] == "Movement data not found for year 2024"


def test_calculate_indirect_alerts_captures_unexpected_errors(indirect_dirs, monkeypatch):
    tmp_path, movement_dir, out_dir = indirect_dirs
    direct_csv = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "direct_alert": ["true"]}).to_csv(direct_csv, index=False)
    _write_movements(movement_dir / "movement_data_base_2024.csv")

    monkeypatch.setattr(
        indirect_alert,
        "pkg_alert_indirect",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("el paquete falló")),
    )

    result = indirect_alert.calculate_indirect_alerts(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_csv=str(direct_csv),
        movement_csv_dir=str(movement_dir),
        output_csv=str(out_dir / "out.csv"),
    )

    assert result["success"] is False
    assert result["error"] == "el paquete falló"


# ===================== calculate_indirect_alerts_batch =====================


def test_calculate_indirect_alerts_batch_aggregates_and_builds_paths(monkeypatch, tmp_path):
    seen = []

    def fake_calculate(**kwargs):
        seen.append(kwargs)
        if kwargs["period"] == "202402":
            return {"success": False, "period": "202402", "error": "sin movimientos"}
        return {
            "success": True,
            "period": kwargs["period"],
            "indirect_alert_in_count": 3,
            "indirect_alert_out_count": 2,
            "enterprise_generated": True,
            "enterprise_entries": 4,
            "enterprise_exits": 1,
            "enterprise_total": 5,
        }

    monkeypatch.setattr(indirect_alert, "calculate_indirect_alerts", fake_calculate)

    result = indirect_alert.calculate_indirect_alerts_batch(
        periods=["202401", "202402"],
        period_type="nad",
        source="smbyc",
        direct_alerts_dir=str(tmp_path / "direct"),
        movement_csv_dir=str(tmp_path / "movements"),
        output_dir=str(tmp_path / "results" / "indirect_alerts"),
    )

    assert result["success"] is False
    assert result["periods_processed"] == 1
    assert result["periods_failed"] == 1
    assert result["total_indirect_in"] == 3
    assert result["total_indirect_out"] == 2
    assert result["enterprise_generated"] is True
    assert result["total_enterprise_entries"] == 4
    assert result["total_enterprise_exits"] == 1
    assert result["total_enterprise_records"] == 5

    # Las rutas se derivan del período siguiendo la convención de nombres.
    assert seen[0]["direct_alerts_csv"].endswith("smbyc_direct_alert_nad_202401.csv")
    assert seen[0]["output_csv"].endswith("smbyc_indirect_alert_nad_202401.csv")
    # Sin enterprise_output_dir, se usa el hermano `enterprise_alerts`.
    assert seen[0]["enterprise_output_csv"].endswith(
        str(tmp_path / "results" / "enterprise_alerts" / "smbyc_enterprise_alert_nad_202401.csv")
    )


def test_calculate_indirect_alerts_batch_honours_explicit_enterprise_dir(monkeypatch, tmp_path):
    seen = []
    monkeypatch.setattr(
        indirect_alert,
        "calculate_indirect_alerts",
        lambda **kwargs: (seen.append(kwargs), {"success": True, "period": kwargs["period"]})[1],
    )

    indirect_alert.calculate_indirect_alerts_batch(
        periods=["2024"],
        period_type="annual",
        source="smbyc",
        direct_alerts_dir="d",
        movement_csv_dir="m",
        output_dir="o",
        enterprise_output_dir=str(tmp_path / "empresas"),
    )

    assert seen[0]["enterprise_output_csv"].startswith(str(tmp_path / "empresas"))


def test_calculate_indirect_alerts_batch_skips_enterprise_path_when_disabled(monkeypatch):
    seen = []
    monkeypatch.setattr(
        indirect_alert,
        "calculate_indirect_alerts",
        lambda **kwargs: (seen.append(kwargs), {"success": True, "period": kwargs["period"]})[1],
    )

    result = indirect_alert.calculate_indirect_alerts_batch(
        periods=["2024"],
        period_type="annual",
        source="smbyc",
        direct_alerts_dir="d",
        movement_csv_dir="m",
        output_dir="o",
        generate_enterprise_alerts=False,
    )

    assert seen[0]["enterprise_output_csv"] is None
    assert result["success"] is True
    assert result["enterprise_generated"] is False
