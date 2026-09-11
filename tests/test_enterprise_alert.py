import pandas as pd
import pytest

import enterprise_alert


# ===================== parse_year_from_period =====================


def test_parse_year_from_period_handles_every_period_type():
    # cumulative usa el último año del rango: los movimientos relevantes son
    # los del cierre del acumulado.
    assert enterprise_alert.parse_year_from_period("2010-2024", "cumulative") == 2024
    assert enterprise_alert.parse_year_from_period("201701", "nad") == 2017
    assert enterprise_alert.parse_year_from_period("202304", "atd") == 2023
    assert enterprise_alert.parse_year_from_period("2017", "annual") == 2017
    # annual con rango cae en el fallback: toma los primeros 4 dígitos.
    assert enterprise_alert.parse_year_from_period("2017-2018", "annual") == 2017
    # cumulative sin guion también cae en el fallback.
    assert enterprise_alert.parse_year_from_period("2020", "cumulative") == 2020


# ===================== load_direct_alerts_with_alert =====================


def test_load_direct_alerts_returns_empty_set_for_missing_file(tmp_path):
    assert enterprise_alert.load_direct_alerts_with_alert(str(tmp_path / "no.csv")) == set()


def test_load_direct_alerts_returns_empty_set_without_id_column(tmp_path):
    csv_path = tmp_path / "direct.csv"
    pd.DataFrame({"otra": ["x"], "direct_alert": [True]}).to_csv(csv_path, index=False)

    assert enterprise_alert.load_direct_alerts_with_alert(str(csv_path)) == set()


def test_load_direct_alerts_returns_empty_set_without_alert_column(tmp_path):
    csv_path = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["123"], "deforested_ha": [1.0]}).to_csv(csv_path, index=False)

    assert enterprise_alert.load_direct_alerts_with_alert(str(csv_path)) == set()


@pytest.mark.parametrize(
    "column",
    ["direct_alert", "intersect_deforestation", "intersect_early_warnings"],
)
def test_load_direct_alerts_accepts_each_supported_alert_column(tmp_path, column):
    csv_path = tmp_path / "direct.csv"
    pd.DataFrame({"id": ["00123", "456"], column: ["true", "false"]}).to_csv(
        csv_path, index=False
    )

    assert enterprise_alert.load_direct_alerts_with_alert(str(csv_path)) == {"123"}


def test_load_direct_alerts_recognises_spanish_and_numeric_truthy_values(tmp_path):
    csv_path = tmp_path / "direct.csv"
    pd.DataFrame(
        {
            "id": ["1", "2", "3", "4", "5", "6", "7"],
            "direct_alert": ["true", "1", "yes", "si", "sí", "false", "0"],
        }
    ).to_csv(csv_path, index=False)

    assert enterprise_alert.load_direct_alerts_with_alert(str(csv_path)) == {
        "1",
        "2",
        "3",
        "4",
        "5",
    }


def test_load_direct_alerts_normalizes_ids_before_returning(tmp_path):
    csv_path = tmp_path / "direct.csv"
    pd.DataFrame(
        {"id": ["FARM_ID_00123", "000123", "456.0"], "direct_alert": ["true"] * 3}
    ).to_csv(csv_path, index=False)

    # Las tres filas colapsan en dos IDs normalizados distintos.
    assert enterprise_alert.load_direct_alerts_with_alert(str(csv_path)) == {"123", "456"}


def test_load_direct_alerts_returns_empty_set_on_unreadable_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "direct.csv"
    csv_path.write_text("id,direct_alert\n1,true\n", encoding="utf-8")

    def boom(*args, **kwargs):
        raise ValueError("csv corrupto")

    monkeypatch.setattr(enterprise_alert.pd, "read_csv", boom)

    assert enterprise_alert.load_direct_alerts_with_alert(str(csv_path)) == set()


# ===================== load_movements_for_year =====================


def _movements_csv(path, rows=None):
    rows = rows or {
        "SIT_CODE_ORIGEN": ["FARM_ID_00123", "999"],
        "SIT_CODE_DESTINO": ["999", "00456"],
        "TIPO_ORIGEN": ["FARM", "SLAUGHTERHOUSE"],
        "TIPO_DESTINO": ["SLAUGHTERHOUSE", "FARM"],
        "PRODUCER_ID_ORIGEN": ["00001", "2"],
        "PRODUCER_ID_DESTINO": ["2", "00001"],
        "DATE": ["2024-01-10", "2024-05-20"],
    }
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_load_movements_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="No existe archivo de movimientos"):
        enterprise_alert.load_movements_for_year(str(tmp_path / "no.csv"))


def test_load_movements_renames_columns_and_normalizes_ids(tmp_path):
    csv_path = _movements_csv(tmp_path / "movement_data_base_2024.csv")

    df = enterprise_alert.load_movements_for_year(str(csv_path))

    assert set(["origen_id", "destination_id", "tipo_origen", "tipo_destino"]) <= set(df.columns)
    assert df["origen_id"].tolist() == ["123", "999"]
    assert df["destination_id"].tolist() == ["999", "456"]
    assert df["producer_id_origen"].tolist() == ["1", "2"]
    assert df["producer_id_destino"].tolist() == ["2", "1"]


def test_load_movements_maps_both_idpro_column_spellings(tmp_path):
    for column in ("IDPRO", "ID_PRO"):
        csv_path = tmp_path / f"movement_{column}.csv"
        pd.DataFrame({"SIT_CODE_ORIGEN": ["1"], column: ["E1"]}).to_csv(csv_path, index=False)

        df = enterprise_alert.load_movements_for_year(str(csv_path))

        assert "idpro" in df.columns
        assert df["idpro"].tolist() == ["E1"]


def test_load_movements_filters_by_quarter(tmp_path):
    csv_path = _movements_csv(tmp_path / "movement_data_base_2024.csv")

    q1 = enterprise_alert.load_movements_for_year(str(csv_path), quarter=1)
    q2 = enterprise_alert.load_movements_for_year(str(csv_path), quarter=2)
    q3 = enterprise_alert.load_movements_for_year(str(csv_path), quarter=3)

    assert q1["origen_id"].tolist() == ["123"]
    assert q2["origen_id"].tolist() == ["999"]
    assert q3.empty


def test_load_movements_keeps_all_rows_when_quarter_filter_fails(tmp_path, monkeypatch):
    csv_path = _movements_csv(tmp_path / "movement_data_base_2024.csv")

    def boom(*args, **kwargs):
        raise ValueError("fecha ilegible")

    monkeypatch.setattr(enterprise_alert.pd, "to_datetime", boom)

    df = enterprise_alert.load_movements_for_year(str(csv_path), quarter=1)

    # El filtro falla pero no se pierde el lote: se devuelven todos los movimientos.
    assert len(df) == 2


def test_load_movements_ignores_quarter_when_there_is_no_date_column(tmp_path):
    csv_path = tmp_path / "movement.csv"
    pd.DataFrame({"SIT_CODE_ORIGEN": ["1", "2"]}).to_csv(csv_path, index=False)

    df = enterprise_alert.load_movements_for_year(str(csv_path), quarter=1)

    assert len(df) == 2


# ===================== load_enterprise_mapping =====================


class _FakeCollection:
    def __init__(self, docs):
        self.docs = docs
        self.last_query = None

    def find(self, query):
        self.last_query = query
        if not query:
            return list(self.docs)
        wanted = query["type_enterprise"]["$in"]
        return [d for d in self.docs if d.get("type_enterprise") in wanted]


class _FakeDb:
    def __init__(self, collection):
        self._collection = collection

    def __getitem__(self, name):
        assert name == enterprise_alert.COL_ENTERPRISE
        return self._collection


class _FakeMongoClient:
    def __init__(self, docs):
        self.collection = _FakeCollection(docs)
        self.closed = False

    def __getitem__(self, name):
        return _FakeDb(self.collection)

    def close(self):
        self.closed = True


def test_load_enterprise_mapping_indexes_by_code(monkeypatch):
    client = _FakeMongoClient(
        [
            {"_id": "id1", "code": "E1", "name": "Frigorífico", "type_enterprise": "SLAUGHTERHOUSE"},
            {"_id": "id2", "code": "E2", "name": "Feria", "type_enterprise": "CATTLE_FAIR"},
            # Sin `code`: se descarta porque no se puede indexar.
            {"_id": "id3", "name": "Sin código", "type_enterprise": "SLAUGHTERHOUSE"},
        ]
    )
    monkeypatch.setattr(enterprise_alert, "MongoClient", lambda uri: client)

    mapping = enterprise_alert.load_enterprise_mapping()

    assert set(mapping) == {"E1", "E2"}
    assert mapping["E1"] == {
        "name": "Frigorífico",
        "type_enterprise": "SLAUGHTERHOUSE",
        "_id": "id1",
    }
    assert client.closed is True


def test_load_enterprise_mapping_filters_by_enterprise_type(monkeypatch):
    client = _FakeMongoClient(
        [
            {"_id": "id1", "code": "E1", "name": "A", "type_enterprise": "SLAUGHTERHOUSE"},
            {"_id": "id2", "code": "E2", "name": "B", "type_enterprise": "CATTLE_FAIR"},
        ]
    )
    monkeypatch.setattr(enterprise_alert, "MongoClient", lambda uri: client)

    mapping = enterprise_alert.load_enterprise_mapping(enterprise_types=["SLAUGHTERHOUSE"])

    assert set(mapping) == {"E1"}
    assert client.collection.last_query == {"type_enterprise": {"$in": ["SLAUGHTERHOUSE"]}}


def test_load_enterprise_mapping_returns_empty_dict_when_mongo_fails(monkeypatch):
    def boom(uri):
        raise ConnectionError("mongo caído")

    monkeypatch.setattr(enterprise_alert, "MongoClient", boom)

    assert enterprise_alert.load_enterprise_mapping() == {}


# ===================== calculate_enterprise_alerts_for_period =====================


@pytest.fixture
def period_dirs(tmp_path):
    direct = tmp_path / "direct"
    movement = tmp_path / "movement"
    output = tmp_path / "out"
    for path in (direct, movement, output):
        path.mkdir(parents=True)
    return direct, movement, output


def _write_direct(direct_dir, period="202401", period_type="nad", ids=("00123",)):
    pd.DataFrame({"id": list(ids), "direct_alert": [True] * len(ids)}).to_csv(
        direct_dir / f"smbyc_direct_alert_{period_type}_{period}.csv", index=False
    )


def test_calculate_enterprise_alerts_writes_csv_and_metadata(period_dirs, monkeypatch):
    direct_dir, movement_dir, output_dir = period_dirs
    _write_direct(direct_dir)
    _movements_csv(movement_dir / "movement_data_base_2024.csv")

    monkeypatch.setattr(
        enterprise_alert,
        "pkg_alert_enterprise",
        lambda **kwargs: pd.DataFrame(
            {
                "idpro": ["E1", "E2"],
                "id_farm": ["123", "123"],
                "typemove": ["in", "out"],
                "enterprise_type": ["SLAUGHTERHOUSE", "CATTLE_FAIR"],
            }
        ),
    )
    monkeypatch.setattr(
        enterprise_alert,
        "load_enterprise_mapping",
        lambda *a, **k: {"E1": {"name": "Frigorífico", "type_enterprise": "SLAUGHTERHOUSE"}},
    )

    result = enterprise_alert.calculate_enterprise_alerts_for_period(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_dir=str(direct_dir),
        movement_csv_dir=str(movement_dir),
        output_dir=str(output_dir),
    )

    assert result["success"] is True
    assert result["year"] == 2024
    assert result["quarter"] == 1
    assert result["entries_count"] == 1
    assert result["exits_count"] == 1
    assert result["unique_enterprises"] == 2
    assert result["unique_farms"] == 1
    assert result["total_records"] == 2

    written = pd.read_csv(
        output_dir / "smbyc_enterprise_alert_nad_202401.csv", dtype={"period": str}
    )
    assert written["period"].tolist() == ["202401", "202401"]
    assert written["year"].tolist() == [2024, 2024]
    assert written["quarter"].tolist() == [1, 1]
    assert written["source"].tolist() == ["smbyc", "smbyc"]
    # El tipo original del paquete se conserva antes de sobrescribirlo con el de BD.
    assert written["enterprise_type_raw"].tolist() == ["SLAUGHTERHOUSE", "CATTLE_FAIR"]
    # La empresa desconocida en BD queda con nombre y tipo vacíos.
    assert written["enterprise_name"].tolist()[0] == "Frigorífico"
    assert pd.isna(written["enterprise_name"].tolist()[1])


def test_calculate_enterprise_alerts_omits_quarter_for_annual_periods(period_dirs, monkeypatch):
    direct_dir, movement_dir, output_dir = period_dirs
    _write_direct(direct_dir, period="2017", period_type="annual")
    _movements_csv(movement_dir / "movement_data_base_2017.csv")

    monkeypatch.setattr(
        enterprise_alert,
        "pkg_alert_enterprise",
        lambda **kwargs: pd.DataFrame(
            {"idpro": ["E1"], "id_farm": ["123"], "typemove": ["in"]}
        ),
    )
    monkeypatch.setattr(enterprise_alert, "load_enterprise_mapping", lambda *a, **k: {})

    result = enterprise_alert.calculate_enterprise_alerts_for_period(
        period="2017",
        period_type="annual",
        source="smbyc",
        direct_alerts_dir=str(direct_dir),
        movement_csv_dir=str(movement_dir),
        output_dir=str(output_dir),
    )

    assert result["success"] is True
    assert result["quarter"] is None
    written = pd.read_csv(output_dir / "smbyc_enterprise_alert_annual_2017.csv")
    assert "quarter" not in written.columns
    # Sin mapeo de empresas no se añaden columnas de enriquecimiento.
    assert "enterprise_name" not in written.columns


def test_calculate_enterprise_alerts_fails_without_alerted_farms(period_dirs):
    direct_dir, movement_dir, output_dir = period_dirs

    result = enterprise_alert.calculate_enterprise_alerts_for_period(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_dir=str(direct_dir),
        movement_csv_dir=str(movement_dir),
        output_dir=str(output_dir),
    )

    assert result == {
        "success": False,
        "period": "202401",
        "error": "No farms with direct alert",
    }


def test_calculate_enterprise_alerts_fails_without_movement_file(period_dirs):
    direct_dir, movement_dir, output_dir = period_dirs
    _write_direct(direct_dir)

    result = enterprise_alert.calculate_enterprise_alerts_for_period(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_dir=str(direct_dir),
        movement_csv_dir=str(movement_dir),
        output_dir=str(output_dir),
    )

    assert result["success"] is False
    assert result["error"] == "Movement data not found for year 2024"


def test_calculate_enterprise_alerts_fails_when_package_returns_nothing(period_dirs, monkeypatch):
    direct_dir, movement_dir, output_dir = period_dirs
    _write_direct(direct_dir)
    _movements_csv(movement_dir / "movement_data_base_2024.csv")

    monkeypatch.setattr(enterprise_alert, "pkg_alert_enterprise", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(enterprise_alert, "load_enterprise_mapping", lambda *a, **k: {})

    result = enterprise_alert.calculate_enterprise_alerts_for_period(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_dir=str(direct_dir),
        movement_csv_dir=str(movement_dir),
        output_dir=str(output_dir),
    )

    assert result["success"] is False
    assert result["error"] == "No movements with alerted farms"


def test_calculate_enterprise_alerts_captures_unexpected_errors(period_dirs, monkeypatch):
    direct_dir, movement_dir, output_dir = period_dirs
    _write_direct(direct_dir)
    _movements_csv(movement_dir / "movement_data_base_2024.csv")

    def boom(**kwargs):
        raise RuntimeError("el paquete falló")

    monkeypatch.setattr(enterprise_alert, "pkg_alert_enterprise", boom)
    monkeypatch.setattr(enterprise_alert, "load_enterprise_mapping", lambda *a, **k: {})

    result = enterprise_alert.calculate_enterprise_alerts_for_period(
        period="202401",
        period_type="nad",
        source="smbyc",
        direct_alerts_dir=str(direct_dir),
        movement_csv_dir=str(movement_dir),
        output_dir=str(output_dir),
    )

    assert result["success"] is False
    assert result["error"] == "el paquete falló"


# ===================== calculate_enterprise_alerts_batch =====================


def test_calculate_enterprise_alerts_batch_aggregates_results(monkeypatch):
    calls = []

    def fake_period(**kwargs):
        calls.append(kwargs["period"])
        if kwargs["period"] == "202402":
            return {"success": False, "period": "202402", "error": "sin datos"}
        return {
            "success": True,
            "period": kwargs["period"],
            "entries_count": 2,
            "exits_count": 3,
            "total_records": 5,
        }

    monkeypatch.setattr(enterprise_alert, "calculate_enterprise_alerts_for_period", fake_period)

    result = enterprise_alert.calculate_enterprise_alerts_batch(
        periods=["202401", "202402", "202403"],
        period_type="nad",
        source="smbyc",
        direct_alerts_dir="d",
        movement_csv_dir="m",
        output_dir="o",
    )

    assert calls == ["202401", "202402", "202403"]
    assert result["success"] is False
    assert result["periods_processed"] == 2
    assert result["periods_failed"] == 1
    assert result["failed_periods"] == ["202402"]
    assert result["total_entries"] == 4
    assert result["total_exits"] == 6
    assert result["total_records"] == 10
    assert len(result["results"]) == 3


def test_calculate_enterprise_alerts_batch_reports_success_when_all_periods_pass(monkeypatch):
    monkeypatch.setattr(
        enterprise_alert,
        "calculate_enterprise_alerts_for_period",
        lambda **kwargs: {"success": True, "period": kwargs["period"], "total_records": 1},
    )

    result = enterprise_alert.calculate_enterprise_alerts_batch(
        periods=["202401"],
        period_type="nad",
        source="smbyc",
        direct_alerts_dir="d",
        movement_csv_dir="m",
        output_dir="o",
    )

    assert result["success"] is True
    assert result["periods_failed"] == 0
    assert result["failed_periods"] == []
