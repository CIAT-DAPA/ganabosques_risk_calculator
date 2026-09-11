import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import main


# ===================== parse_steps =====================


def test_parse_steps_supports_single_values_and_ranges():
    assert main.parse_steps(["1"]) == [1]
    assert main.parse_steps(["1-3"]) == [1, 2, 3]
    assert main.parse_steps(["1-3", "5"]) == [1, 2, 3, 5]
    assert main.parse_steps(["1", "3", "5"]) == [1, 3, 5]
    # Los duplicados colapsan y el resultado sale ordenado.
    assert main.parse_steps(["3", "1", "3"]) == [1, 3]


def test_parse_steps_all_expands_to_every_step():
    assert main.parse_steps(["all"]) == list(range(1, 10))
    assert main.parse_steps(["ALL"]) == list(range(1, 10))
    # 'all' corta el resto de la expresión.
    assert main.parse_steps(["2", "all"]) == list(range(1, 10))


def test_parse_steps_returns_empty_for_no_arguments():
    assert main.parse_steps([]) == []
    assert main.parse_steps(None) == []


def test_parse_steps_discards_out_of_range_and_invalid_tokens(capsys):
    assert main.parse_steps(["0", "10", "5"]) == [5]
    assert main.parse_steps(["abc"]) == []
    assert main.parse_steps(["a-b"]) == []

    salida = capsys.readouterr().out
    assert "Paso inválido" in salida
    assert "Formato de paso inválido" in salida


# ===================== connect_mongodb =====================


def test_connect_mongodb_returns_true_on_success(monkeypatch):
    llamadas = {}
    monkeypatch.setattr(
        main, "connect", lambda host, db: llamadas.update(host=host, db=db), raising=False
    )
    monkeypatch.setenv("MONGO_URI", "mongodb://ejemplo:27017")
    monkeypatch.setenv("MONGO_DB_NAME", "ganabosques_test")

    assert main.connect_mongodb() is True
    assert llamadas == {"host": "mongodb://ejemplo:27017", "db": "ganabosques_test"}


def test_connect_mongodb_uses_defaults(monkeypatch):
    llamadas = {}
    monkeypatch.delenv("MONGO_URI", raising=False)
    monkeypatch.delenv("MONGO_DB_NAME", raising=False)
    monkeypatch.setattr(
        main, "connect", lambda host, db: llamadas.update(host=host, db=db), raising=False
    )

    assert main.connect_mongodb() is True
    assert llamadas == {"host": "mongodb://localhost:27017", "db": "ganabosques"}


def test_connect_mongodb_returns_false_on_error(monkeypatch):
    monkeypatch.setattr(
        main,
        "connect",
        lambda host, db: (_ for _ in ()).throw(RuntimeError("sin red")),
        raising=False,
    )

    assert main.connect_mongodb() is False


# ===================== validate_prerequisites =====================


@pytest.fixture
def workspace(tmp_path):
    base = tmp_path / "alertas"
    (base / "movements").mkdir(parents=True)
    (base / "results" / "smbyc" / "annual" / "direct_alerts").mkdir(parents=True)
    return base


class FakeDataManager:
    def __init__(self, geojsons_dir):
        self.geojsons_dir = Path(geojsons_dir)


def test_validate_prerequisites_requires_periods():
    ok, error = main.validate_prerequisites("direct", None, None, [])

    assert ok is False
    assert "No hay períodos" in error


def test_validate_direct_stage_accepts_data_manager_and_metadata(tmp_path):
    dm = FakeDataManager(tmp_path)

    ok, error = main.validate_prerequisites("direct", dm, [{"sitcode": "A1"}], ["2024"])

    assert (ok, error) == (True, "")


def test_validate_direct_stage_requires_data_manager():
    ok, error = main.validate_prerequisites("direct", None, [{"sitcode": "A1"}], ["2024"])

    assert ok is False
    assert "DataManager no disponible" in error


def test_validate_direct_stage_accepts_local_geojsons(tmp_path):
    dm = FakeDataManager(tmp_path)
    (tmp_path / "A1.geojson").write_text("{}", encoding="utf-8")

    ok, error = main.validate_prerequisites("direct", dm, None, ["2024"])

    assert (ok, error) == (True, "")


def test_validate_direct_stage_rejects_empty_geojson_folder(tmp_path):
    dm = FakeDataManager(tmp_path)

    ok, error = main.validate_prerequisites("direct", dm, None, ["2024"])

    assert ok is False
    assert "No hay geojsons disponibles" in error


def test_validate_direct_stage_rejects_missing_geojson_folder(tmp_path):
    dm = FakeDataManager(tmp_path / "no_existe")

    ok, error = main.validate_prerequisites("direct", dm, None, ["2024"])

    assert ok is False
    assert "no se encontró carpeta de geojsons" in error


def test_validate_direct_stage_offline_requires_configured_folder():
    ok, error = main.validate_prerequisites(
        "direct", None, None, ["2024"], offline_mode=True, geojsons_folder=""
    )

    assert ok is False
    assert "FOLDER_GEOJSONS no está configurado" in error


def test_validate_direct_stage_offline_requires_existing_folder(tmp_path):
    ok, error = main.validate_prerequisites(
        "direct", None, None, ["2024"], offline_mode=True,
        geojsons_folder=str(tmp_path / "no_existe"),
    )

    assert ok is False
    assert "carpeta de geojsons no existe" in error


def test_validate_direct_stage_offline_requires_geojson_files(tmp_path):
    ok, error = main.validate_prerequisites(
        "direct", None, None, ["2024"], offline_mode=True, geojsons_folder=str(tmp_path)
    )

    assert ok is False
    assert "no hay .geojson" in error


def test_validate_direct_stage_offline_accepts_folder_with_files(tmp_path):
    (tmp_path / "A1.geojson").write_text("{}", encoding="utf-8")

    ok, error = main.validate_prerequisites(
        "direct", None, None, ["2024"], offline_mode=True, geojsons_folder=str(tmp_path)
    )

    assert (ok, error) == (True, "")


def test_validate_movement_stage_requires_movements_and_direct_alerts(workspace):
    (workspace / "movements" / "movement_data_base_2024.csv").write_text("x", encoding="utf-8")

    ok, error = main.validate_prerequisites(
        "movement", None, None, ["2024"], workspace_dir=workspace,
        source="smbyc", period_type="annual",
    )

    assert (ok, error) == (True, "")


def test_validate_movement_stage_reports_missing_movement_files(workspace):
    ok, error = main.validate_prerequisites(
        "movement", None, None, ["2024"], workspace_dir=workspace,
        source="smbyc", period_type="annual",
    )

    assert ok is False
    assert "No hay archivos de movimientos" in error


def test_validate_movement_stage_reports_missing_movements_dir(tmp_path):
    base = tmp_path / "alertas"
    (base / "results" / "smbyc" / "annual" / "direct_alerts").mkdir(parents=True)

    ok, error = main.validate_prerequisites(
        "movement", None, None, ["2024"], workspace_dir=base,
        source="smbyc", period_type="annual",
    )

    assert ok is False
    assert "Directorio de movimientos no existe" in error


def test_validate_movement_stage_reports_missing_direct_alerts(tmp_path):
    base = tmp_path / "alertas"
    (base / "movements").mkdir(parents=True)
    (base / "movements" / "movement_data_base_2024.csv").write_text("x", encoding="utf-8")

    ok, error = main.validate_prerequisites(
        "movement", None, None, ["2024"], workspace_dir=base,
        source="smbyc", period_type="annual",
    )

    assert ok is False
    assert "No hay alertas directas previas" in error


def test_validate_metrics_stage(tmp_path):
    dm = FakeDataManager(tmp_path)

    assert main.validate_prerequisites("metrics", dm, [{"sitcode": "A1"}], ["2024"]) == (True, "")

    ok, error = main.validate_prerequisites("metrics", None, None, ["2024"])
    assert ok is False
    assert "DataManager requerido" in error


def test_validate_metrics_stage_accepts_local_geojsons(tmp_path):
    dm = FakeDataManager(tmp_path)
    (tmp_path / "A1.geojson").write_text("{}", encoding="utf-8")

    assert main.validate_prerequisites("metrics", dm, None, ["2024"]) == (True, "")


def test_validate_metrics_stage_rejects_empty_folder(tmp_path):
    dm = FakeDataManager(tmp_path)

    ok, error = main.validate_prerequisites("metrics", dm, None, ["2024"])

    assert ok is False
    assert "No hay geojsons disponibles" in error


def test_validate_metrics_stage_rejects_missing_folder(tmp_path):
    dm = FakeDataManager(tmp_path / "no_existe")

    ok, error = main.validate_prerequisites("metrics", dm, None, ["2024"])

    assert ok is False
    assert "no se encontró carpeta de geojsons" in error


def test_validate_enterprise_stage(workspace):
    (workspace / "movements" / "movement_data_base_2024.csv").write_text("x", encoding="utf-8")

    assert main.validate_prerequisites(
        "enterprise", None, None, ["2024"], workspace_dir=workspace,
        source="smbyc", period_type="annual",
    ) == (True, "")


def test_validate_enterprise_stage_reports_missing_movements(workspace):
    ok, error = main.validate_prerequisites(
        "enterprise", None, None, ["2024"], workspace_dir=workspace,
        source="smbyc", period_type="annual",
    )

    assert ok is False
    assert "No hay archivos de movimientos" in error


def test_validate_enterprise_stage_reports_missing_movements_dir(tmp_path):
    base = tmp_path / "alertas"
    (base / "results" / "smbyc" / "annual" / "direct_alerts").mkdir(parents=True)

    ok, error = main.validate_prerequisites(
        "enterprise", None, None, ["2024"], workspace_dir=base,
        source="smbyc", period_type="annual",
    )

    assert ok is False
    assert "Directorio de movimientos no existe" in error


def test_validate_total_stage_requires_direct_alerts(workspace):
    assert main.validate_prerequisites(
        "total", None, None, ["2024"], workspace_dir=workspace,
        source="smbyc", period_type="annual",
    ) == (True, "")


def test_validate_total_stage_reports_missing_direct_alerts(tmp_path):
    base = tmp_path / "alertas"
    base.mkdir(parents=True)

    ok, error = main.validate_prerequisites(
        "total", None, None, ["2024"], workspace_dir=base,
        source="smbyc", period_type="annual",
    )

    assert ok is False
    assert "No hay alertas directas" in error


def test_validate_total_stage_warns_about_missing_metrics(workspace, caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        ok, _ = main.validate_prerequisites(
            "total", None, None, ["2024"], workspace_dir=workspace,
            source="smbyc", period_type="annual",
        )

    assert ok is True
    assert "No hay métricas espaciales" in caplog.text


def test_print_validation_summary(capsys):
    main.print_validation_summary("direct", True, "")
    main.print_validation_summary("direct", False, "falta el raster")

    salida = capsys.readouterr().out
    assert "Validación direct: OK" in salida
    assert "falta el raster" in salida


# ===================== query_available_periods =====================


class FakeEnum:
    def __init__(self, value):
        self.value = value


@pytest.fixture
def deforestation_orm(monkeypatch):
    class FakeDeforestation:
        store = []

        class objects:
            pass

    periodos = []

    class Manager:
        def __call__(self, **filters):
            class QS:
                def order_by(self, *a):
                    return periodos

            return QS()

    FakeDeforestation.objects = Manager()

    source_enum = SimpleNamespace(SMBYC=FakeEnum("smbyc"))
    type_enum = SimpleNamespace(
        ANNUAL=FakeEnum("annual"),
        CUMULATIVE=FakeEnum("cumulative"),
        NAD=FakeEnum("nad"),
        ATD=FakeEnum("atd"),
    )

    monkeypatch.setattr(main, "HAS_ORM", True)
    monkeypatch.setattr(main, "Deforestation", FakeDeforestation, raising=False)
    monkeypatch.setattr(
        main, "DeforestationSource", {"SMBYC": source_enum.SMBYC}, raising=False
    )
    monkeypatch.setattr(main, "DeforestationType", type_enum, raising=False)
    return periodos


def _periodo(nombre, inicio=2013, fin=2014, oid="oid1"):
    return SimpleNamespace(
        period_start=datetime(inicio, 1, 1),
        period_end=datetime(fin, 12, 31),
        name=nombre,
        id=oid,
    )


def test_query_available_periods_returns_none_without_orm(monkeypatch, capsys):
    monkeypatch.setattr(main, "HAS_ORM", False)

    assert main.query_available_periods("smbyc", "annual") == []
    assert "ORM no disponible" in capsys.readouterr().out


def test_query_available_periods_returns_tuples(deforestation_orm):
    deforestation_orm.append(_periodo("smbyc_deforestation_annual_2013-2014"))

    resultado = main.query_available_periods("smbyc", "annual")

    assert len(resultado) == 1
    assert resultado[0][2] == "smbyc_deforestation_annual_2013-2014"
    assert resultado[0][3] == "oid1"


def test_query_available_periods_skips_incomplete_records(deforestation_orm):
    incompleto = _periodo("capa_sin_fechas")
    incompleto.period_start = None
    deforestation_orm.append(incompleto)

    assert main.query_available_periods("smbyc", "annual") == []


def test_query_available_periods_rejects_unknown_type(deforestation_orm, capsys):
    assert main.query_available_periods("smbyc", "inexistente") == []
    assert "Tipo de deforestación desconocido" in capsys.readouterr().out


def test_query_available_periods_reports_unknown_source(deforestation_orm, capsys):
    assert main.query_available_periods("fuente_rara", "annual") == []
    assert "Fuente o tipo no válido" in capsys.readouterr().out


def test_query_available_periods_reports_query_errors(deforestation_orm, monkeypatch, capsys):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    monkeypatch.setattr(main.Deforestation, "objects", Manager())

    assert main.query_available_periods("smbyc", "annual") == []
    assert "Error consultando períodos" in capsys.readouterr().out


# ===================== generate_year_ranges =====================


def _tupla(nombre, inicio=2013, fin=2014, oid="oid1"):
    """generate_year_ranges consume tuplas (inicio, fin, nombre, mongo_id)."""
    return (datetime(inicio, 1, 1), datetime(fin, 12, 31), nombre, oid)


def test_generate_year_ranges_returns_empty_without_layers(capsys):
    assert main.generate_year_ranges("2024", "smbyc", "annual", []) == []
    assert "No hay capas" in capsys.readouterr().out


def test_generate_year_ranges_annual():
    disponibles = [
        _tupla("smbyc_deforestation_annual_2013-2014"),
        _tupla("smbyc_deforestation_annual_2014-2015"),
        _tupla("smbyc_deforestation_annual_2020-2021"),
    ]

    assert main.generate_year_ranges("2013-2015", "smbyc", "annual", disponibles) == [
        "2013-2014",
        "2014-2015",
    ]


def test_generate_year_ranges_annual_single_year():
    disponibles = [_tupla("smbyc_deforestation_annual_2013-2014")]

    assert main.generate_year_ranges("2013", "smbyc", "annual", disponibles) == ["2013-2014"]


def test_generate_year_ranges_cumulative():
    disponibles = [
        _tupla("smbyc_deforestation_cumulative_2010-2013"),
        _tupla("smbyc_deforestation_cumulative_2010-2020"),
    ]

    assert main.generate_year_ranges("2010-2015", "smbyc", "cumulative", disponibles) == [
        "2010-2013"
    ]


@pytest.mark.parametrize("tipo", ["nad", "atd"])
def test_generate_year_ranges_quarterly(tipo):
    disponibles = [
        _tupla(f"smbyc_deforestation_{tipo}_201701"),
        _tupla(f"smbyc_deforestation_{tipo}_201702"),
        # Repetido: no debe duplicarse.
        _tupla(f"otra_capa_{tipo}_201702"),
        _tupla(f"smbyc_deforestation_{tipo}_202001"),
    ]

    assert main.generate_year_ranges("2017", "smbyc", tipo, disponibles) == ["201701", "201702"]


def test_generate_year_ranges_ignores_names_without_period_code():
    disponibles = [_tupla("capa_sin_codigo")]

    assert main.generate_year_ranges("2024", "smbyc", "annual", disponibles) == []
    assert main.generate_year_ranges("2024", "smbyc", "cumulative", disponibles) == []
    assert main.generate_year_ranges("2024", "smbyc", "nad", disponibles) == []


# ===================== run_direct_alerts =====================


@pytest.fixture
def direct_params(tmp_path):
    return {
        "source": "smbyc",
        "period_type": "annual",
        "years": ["2013-2014"],
        "farm_folder": str(tmp_path / "farms"),
        "raster_template": "",
        "output_csv": str(tmp_path / "out.csv"),
        "_workspace_dir": str(tmp_path),
    }


class FakeDM:
    def __init__(self, workspace_dir):
        self.workspace_dir = Path(workspace_dir)
        self.geojsons_dir = self.workspace_dir / "geojsons"
        self.results_dir = self.workspace_dir / "results"
        self.rasters_ok = True

    def extract_time_filter_from_name(self, nombre):
        return "2013-2014"

    def ensure_raster_available(self, **kwargs):
        return str(self.workspace_dir / "raster.tif") if self.rasters_ok else None

    def get_mongo_id_map_df(self):
        return pd.DataFrame({"id": ["1"], "farm_id": ["f1"], "farm_poligons_id": ["p1"]})


def test_run_direct_alerts_prepares_rasters_from_database(direct_params, monkeypatch, tmp_path):
    dm = FakeDM(tmp_path)
    direct_params["_data_manager"] = dm
    direct_params["_available_periods"] = [
        (datetime(2013, 1, 1), datetime(2014, 1, 1), "smbyc_deforestation_annual_2013-2014", "oid1")
    ]

    monkeypatch.setattr(
        main, "Deforestation",
        SimpleNamespace(objects=lambda **k: SimpleNamespace(first=lambda: SimpleNamespace(path="capa"))),
        raising=False,
    )
    capturado = {}
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: (capturado.update(kwargs), {"periods_processed": 1, "farms_processed": 5, "execution_time": 1.0})[1],
    )

    assert main.run_direct_alerts("Paso 1", False, direct_params, []) is True
    assert capturado["raster_paths_dict"] == {"2013-2014": str(tmp_path / "raster.tif")}


def test_run_direct_alerts_reports_periods_without_metadata(direct_params, monkeypatch, tmp_path, capsys):
    direct_params["_data_manager"] = FakeDM(tmp_path)
    direct_params["_available_periods"] = [
        (datetime(2020, 1, 1), datetime(2021, 1, 1), "otra_capa_2020-2021", "oid9")
    ]
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: {"periods_processed": 0, "farms_processed": 0, "execution_time": 0.0},
    )

    assert main.run_direct_alerts("Paso 1", False, direct_params, []) is True
    assert "No se encontró metadata para período" in capsys.readouterr().out


def test_run_direct_alerts_reports_layers_without_path(direct_params, monkeypatch, tmp_path, capsys):
    direct_params["_data_manager"] = FakeDM(tmp_path)
    direct_params["_available_periods"] = [
        (datetime(2013, 1, 1), datetime(2014, 1, 1), "smbyc_deforestation_annual_2013-2014", "oid1")
    ]
    monkeypatch.setattr(
        main, "Deforestation",
        SimpleNamespace(objects=lambda **k: SimpleNamespace(first=lambda: None)),
        raising=False,
    )
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: {"periods_processed": 0, "farms_processed": 0, "execution_time": 0.0},
    )

    main.run_direct_alerts("Paso 1", False, direct_params, [])

    assert "No se encontró path" in capsys.readouterr().out


def test_run_direct_alerts_reports_raster_download_failures(direct_params, monkeypatch, tmp_path, capsys):
    dm = FakeDM(tmp_path)
    dm.rasters_ok = False
    direct_params["_data_manager"] = dm
    direct_params["_available_periods"] = [
        (datetime(2013, 1, 1), datetime(2014, 1, 1), "smbyc_deforestation_annual_2013-2014", "oid1")
    ]
    monkeypatch.setattr(
        main, "Deforestation",
        SimpleNamespace(objects=lambda **k: SimpleNamespace(first=lambda: SimpleNamespace(path="capa"))),
        raising=False,
    )
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: {"periods_processed": 0, "farms_processed": 0, "execution_time": 0.0},
    )

    main.run_direct_alerts("Paso 1", False, direct_params, [])

    assert "No se pudo obtener raster" in capsys.readouterr().out


def test_run_direct_alerts_reports_layer_lookup_errors(direct_params, monkeypatch, tmp_path, capsys):
    direct_params["_data_manager"] = FakeDM(tmp_path)
    direct_params["_available_periods"] = [
        (datetime(2013, 1, 1), datetime(2014, 1, 1), "smbyc_deforestation_annual_2013-2014", "oid1")
    ]

    def explota(**kwargs):
        raise RuntimeError("mongo caído")

    monkeypatch.setattr(main, "Deforestation", SimpleNamespace(objects=explota), raising=False)
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: {"periods_processed": 0, "farms_processed": 0, "execution_time": 0.0},
    )

    main.run_direct_alerts("Paso 1", False, direct_params, [])

    assert "Error obteniendo path" in capsys.readouterr().out


def test_run_direct_alerts_falls_back_to_local_rasters(direct_params, monkeypatch, tmp_path, capsys):
    dm = FakeDM(tmp_path)
    direct_params["_data_manager"] = dm
    direct_params["_available_periods"] = []
    carpeta = dm.workspace_dir / "rasters" / "smbyc" / "annual"
    carpeta.mkdir(parents=True)
    (carpeta / "smbyc_deforestation_annual_2013-2014.tif").write_bytes(b"x")

    capturado = {}
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: (capturado.update(kwargs), {"periods_processed": 1, "farms_processed": 1, "execution_time": 0.1})[1],
    )

    assert main.run_direct_alerts("Paso 1", False, direct_params, []) is True
    assert "Raster encontrado" in capsys.readouterr().out
    assert capturado["raster_paths_dict"]


def test_run_direct_alerts_reports_missing_raster_folder(direct_params, monkeypatch, tmp_path, capsys):
    direct_params["_data_manager"] = FakeDM(tmp_path)
    direct_params["_available_periods"] = []
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: {"periods_processed": 0, "farms_processed": 0, "execution_time": 0.0},
    )

    main.run_direct_alerts("Paso 1", False, direct_params, [])

    assert "Carpeta de rasters no existe" in capsys.readouterr().out


def test_run_direct_alerts_reports_rasters_not_found_in_folder(direct_params, monkeypatch, tmp_path, capsys):
    dm = FakeDM(tmp_path)
    direct_params["_data_manager"] = dm
    direct_params["_available_periods"] = []
    (dm.workspace_dir / "rasters" / "smbyc" / "annual").mkdir(parents=True)
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: {"periods_processed": 0, "farms_processed": 0, "execution_time": 0.0},
    )

    main.run_direct_alerts("Paso 1", False, direct_params, [])

    salida = capsys.readouterr().out
    assert "Raster no encontrado" in salida
    assert "No se encontraron rasters" in salida


def test_run_direct_alerts_without_data_manager(direct_params, monkeypatch):
    capturado = {}
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: (capturado.update(kwargs), {"periods_processed": 1, "farms_processed": 1, "execution_time": 0.1})[1],
    )

    assert main.run_direct_alerts("Paso 1", False, direct_params, []) is True
    assert capturado["raster_paths_dict"] is None


def test_run_direct_alerts_parallel_mode(direct_params, monkeypatch, capsys):
    direct_params["_use_parallel"] = True
    direct_params["_num_workers"] = 4
    monkeypatch.setattr(
        main, "run_parallel_direct_alerts",
        lambda params, num_workers, cleanup: {
            "num_workers": num_workers, "chunks_processed": 4,
            "periods_processed": 1, "execution_time": 12.0,
        },
    )

    assert main.run_direct_alerts("Paso 1", False, direct_params, []) is True
    assert "Usando procesamiento PARALELO" in capsys.readouterr().out


def test_run_direct_alerts_handles_errors(direct_params, monkeypatch):
    monkeypatch.setattr(
        main, "calculate_direct_alerts",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("cálculo roto")),
    )

    assert main.run_direct_alerts("Paso 1", False, direct_params, []) is False
    # Con --continue-on-error la etapa se reporta como superada.
    assert main.run_direct_alerts("Paso 1", True, direct_params, []) is True


# ===================== run_spatial_metrics =====================


def test_run_spatial_metrics_succeeds(direct_params, monkeypatch, tmp_path):
    dm = FakeDM(tmp_path)
    direct_params["_data_manager"] = dm
    direct_params["_farms_metadata"] = [{"sitcode": "A1"}]

    import spatial_metrics

    monkeypatch.setattr(
        spatial_metrics, "calculate_spatial_metrics",
        lambda **kwargs: {"success": True, "farms_processed": 1, "output_file": "m.csv", "execution_time": 1.0},
    )

    assert main.run_spatial_metrics("Paso 3", False, direct_params) is True


def test_run_spatial_metrics_requires_data_manager(direct_params, capsys):
    assert main.run_spatial_metrics("Paso 3", False, direct_params) is False
    assert "DataManager no disponible" in capsys.readouterr().out


def test_run_spatial_metrics_accepts_local_geojsons(direct_params, monkeypatch, tmp_path, capsys):
    dm = FakeDM(tmp_path)
    dm.geojsons_dir.mkdir(parents=True)
    (dm.geojsons_dir / "A1.geojson").write_text("{}", encoding="utf-8")
    direct_params["_data_manager"] = dm
    direct_params["_farms_metadata"] = None

    import spatial_metrics

    monkeypatch.setattr(
        spatial_metrics, "calculate_spatial_metrics",
        lambda **kwargs: {"success": True, "farms_processed": 1, "output_file": "m.csv", "execution_time": 1.0},
    )

    assert main.run_spatial_metrics("Paso 3", False, direct_params) is True
    assert "Modo offline/mixto" in capsys.readouterr().out


def test_run_spatial_metrics_requires_geometries(direct_params, tmp_path, capsys):
    direct_params["_data_manager"] = FakeDM(tmp_path)
    direct_params["_farms_metadata"] = None

    assert main.run_spatial_metrics("Paso 3", False, direct_params) is False
    assert "No hay metadata de farms ni GeoJSONs" in capsys.readouterr().out


def test_run_spatial_metrics_reports_failure(direct_params, monkeypatch, tmp_path, capsys):
    direct_params["_data_manager"] = FakeDM(tmp_path)
    direct_params["_farms_metadata"] = [{"sitcode": "A1"}]

    import spatial_metrics

    monkeypatch.setattr(spatial_metrics, "calculate_spatial_metrics", lambda **k: {"success": False})

    assert main.run_spatial_metrics("Paso 3", False, direct_params) is False
    assert "Error calculando métricas" in capsys.readouterr().out


def test_run_spatial_metrics_handles_exceptions(direct_params, monkeypatch, tmp_path):
    direct_params["_data_manager"] = FakeDM(tmp_path)
    direct_params["_farms_metadata"] = [{"sitcode": "A1"}]

    import spatial_metrics

    monkeypatch.setattr(
        spatial_metrics, "calculate_spatial_metrics",
        lambda **k: (_ for _ in ()).throw(RuntimeError("métricas rotas")),
    )

    assert main.run_spatial_metrics("Paso 3", False, direct_params) is False
    assert main.run_spatial_metrics("Paso 3", True, direct_params) is True


def test_run_spatial_metrics_handles_import_errors(direct_params, monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "spatial_metrics", None)

    assert main.run_spatial_metrics("Paso 3", False, direct_params) is False
    assert main.run_spatial_metrics("Paso 3", True, direct_params) is True


# ===================== run_indirect_alerts / run_enterprise_alerts =====================


def test_run_indirect_alerts_reports_statistics(direct_params, monkeypatch, capsys):
    monkeypatch.setattr(
        main, "calculate_indirect_alerts_batch",
        lambda **kwargs: {
            "success": True, "periods_processed": 1,
            "total_indirect_in": 3, "total_indirect_out": 2,
            "enterprise_generated": True, "total_enterprise_entries": 4,
            "total_enterprise_exits": 1, "total_enterprise_records": 5,
        },
    )

    assert main.run_indirect_alerts("Paso 2", False, direct_params) is True
    salida = capsys.readouterr().out
    assert "Alertas indirectas (IN): 3" in salida
    assert "Alertas de empresa generadas" in salida


def test_run_indirect_alerts_without_enterprise_output(direct_params, monkeypatch, capsys):
    monkeypatch.setattr(
        main, "calculate_indirect_alerts_batch",
        lambda **kwargs: {
            "success": True, "periods_processed": 1,
            "total_indirect_in": 0, "total_indirect_out": 0,
            "enterprise_generated": False,
        },
    )

    assert main.run_indirect_alerts("Paso 2", False, direct_params) is True
    assert "Alertas de empresa generadas" not in capsys.readouterr().out


def test_run_indirect_alerts_reports_failures(direct_params, monkeypatch):
    monkeypatch.setattr(
        main, "calculate_indirect_alerts_batch",
        lambda **kwargs: {"success": False, "periods_processed": 0, "periods_failed": 1},
    )

    assert main.run_indirect_alerts("Paso 2", False, direct_params) is True
    assert main.run_indirect_alerts("Paso 2", True, direct_params) is False


def test_run_indirect_alerts_handles_exceptions(direct_params, monkeypatch):
    monkeypatch.setattr(
        main, "calculate_indirect_alerts_batch",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("indirectas rotas")),
    )

    assert main.run_indirect_alerts("Paso 2", False, direct_params) is False
    assert main.run_indirect_alerts("Paso 2", True, direct_params) is True


def test_run_enterprise_alerts_reports_statistics(direct_params, monkeypatch, capsys):
    monkeypatch.setattr(
        main, "calculate_enterprise_alerts_batch",
        lambda **kwargs: {
            "success": True, "periods_processed": 1,
            "total_entries": 2, "total_exits": 1, "total_records": 3,
        },
    )

    assert main.run_enterprise_alerts("Paso 4", False, direct_params) is True
    assert "Movimientos entrada (in): 2" in capsys.readouterr().out


def test_run_enterprise_alerts_reports_failed_periods(direct_params, monkeypatch, capsys):
    monkeypatch.setattr(
        main, "calculate_enterprise_alerts_batch",
        lambda **kwargs: {
            "success": False, "periods_processed": 0, "periods_failed": 1,
            "failed_periods": ["2024"],
        },
    )

    assert main.run_enterprise_alerts("Paso 4", False, direct_params) is True
    assert "Períodos con error: 2024" in capsys.readouterr().out


def test_run_enterprise_alerts_handles_exceptions(direct_params, monkeypatch):
    monkeypatch.setattr(
        main, "calculate_enterprise_alerts_batch",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("empresas rotas")),
    )

    assert main.run_enterprise_alerts("Paso 4", False, direct_params) is False
    assert main.run_enterprise_alerts("Paso 4", True, direct_params) is True


# ===================== run_total_risk =====================


def test_run_total_risk_succeeds(direct_params, monkeypatch, tmp_path, capsys):
    direct_params["_data_manager"] = FakeDM(tmp_path)

    import total_alert

    capturado = {}
    monkeypatch.setattr(
        total_alert, "calculate_total_risk",
        lambda **kwargs: (capturado.update(kwargs), {
            "success": True, "periods_processed": 1,
            "files_generated": ["t.csv"], "execution_time": 2.0,
        })[1],
    )

    assert main.run_total_risk("Paso 5", False, direct_params) is True
    assert capturado["mongo_map_df"] is not None
    assert "Mapeo MongoDB disponible" in capsys.readouterr().out


def test_run_total_risk_without_data_manager(direct_params, monkeypatch):
    import total_alert

    monkeypatch.setattr(
        total_alert, "calculate_total_risk",
        lambda **kwargs: {"success": True, "periods_processed": 1, "files_generated": ["t.csv"], "execution_time": 1.0},
    )

    assert main.run_total_risk("Paso 5", False, direct_params) is True


def test_run_total_risk_requires_workspace(monkeypatch, capsys):
    assert main.run_total_risk("Paso 5", False, {}) is False
    assert "workspace_dir no disponible" in capsys.readouterr().out


def test_run_total_risk_reports_failure(direct_params, monkeypatch, capsys):
    import total_alert

    monkeypatch.setattr(total_alert, "calculate_total_risk", lambda **k: {"success": False})

    assert main.run_total_risk("Paso 5", False, direct_params) is False
    assert "No se generaron archivos" in capsys.readouterr().out


def test_run_total_risk_handles_exceptions(direct_params, monkeypatch):
    import total_alert

    monkeypatch.setattr(
        total_alert, "calculate_total_risk",
        lambda **k: (_ for _ in ()).throw(RuntimeError("total roto")),
    )

    assert main.run_total_risk("Paso 5", False, direct_params) is False
    assert main.run_total_risk("Paso 5", True, direct_params) is True


# ===================== run_save_to_db =====================


class SaveDM:
    def __init__(self, base, total_rows=1, enterprise_rows=1):
        self.base = Path(base)
        self.results_dir = self.base / "results"
        self._total_rows = total_rows
        self._enterprise_rows = enterprise_rows
        self.analysis_error = None
        self.saved_bulk = False
        self.farm_result = (1, 0, [])
        self.enterprise_result = (1, 0, [])

    def create_or_get_analysis(self, deforestation_id, value_chain="livestock"):
        if self.analysis_error:
            return (None, self.analysis_error)
        return ("507f1f77bcf86cd799439011", None)

    def get_results_dir(self, stage, source=None, deforestation_type=None, period=None):
        ruta = self.results_dir / (source or "") / (deforestation_type or "") / stage
        ruta.mkdir(parents=True, exist_ok=True)
        return ruta

    def save_farm_risk_to_db(self, df, analysis_id):
        return self.farm_result

    def save_farm_risk_to_db_bulk(self, df, analysis_id, chunk_size=1000):
        self.saved_bulk = True
        return self.farm_result

    def get_farmrisk_cache_for_analysis(self, analysis_id):
        return {}

    def save_enterprise_risk_to_db(self, df, analysis_id, farm_risk_map=None):
        return self.enterprise_result


def _save_params(dm, **overrides):
    params = {
        "data_manager": dm,
        "available_periods": [
            (datetime(2024, 1, 1), datetime(2024, 3, 31), "smbyc_deforestation_nad_202401", "defo1")
        ],
        "periods": ["202401"],
        "source": "smbyc",
        "period_type": "nad",
        "value_chain": "livestock",
        "bulk_insert": False,
        "chunk_size": 1000,
        "args": SimpleNamespace(),
    }
    params.update(overrides)
    return params


def _write_total_and_enterprise(dm, source="smbyc", period_type="nad", period="202401"):
    total_dir = dm.get_results_dir("total_risk", source=source, deforestation_type=period_type)
    pd.DataFrame({"id": ["S1"], "direct_alert": [True]}).to_csv(
        total_dir / f"{source}_total_risk_{period_type}_{period}.csv", index=False
    )
    ent_dir = dm.get_results_dir("enterprise_alerts", source=source, deforestation_type=period_type)
    pd.DataFrame({"idpro": ["E1"], "id_farm": ["S1"], "typemove": ["in"]}).to_csv(
        ent_dir / f"{source}_enterprise_alert_{period_type}_{period}.csv", index=False
    )


def test_run_save_to_db_saves_both_collections(tmp_path, capsys):
    dm = SaveDM(tmp_path)
    _write_total_and_enterprise(dm)

    assert main.run_save_to_db("Paso 6", False, _save_params(dm)) is True
    salida = capsys.readouterr().out
    assert "FarmRisk: 1 guardados" in salida
    assert "EnterpriseRisk: 1 guardados" in salida


def test_run_save_to_db_uses_bulk_mode(tmp_path, capsys):
    dm = SaveDM(tmp_path)
    _write_total_and_enterprise(dm)

    main.run_save_to_db("Paso 6", False, _save_params(dm, bulk_insert=True, chunk_size=500))

    assert dm.saved_bulk is True
    assert "Modo BULK INSERT activado" in capsys.readouterr().out


def test_run_save_to_db_requires_data_manager(capsys):
    assert main.run_save_to_db("Paso 6", False, {"data_manager": None}) is False
    assert "DataManager no disponible" in capsys.readouterr().out


def test_run_save_to_db_skips_periods_without_layer(tmp_path, monkeypatch, capsys):
    dm = SaveDM(tmp_path)
    monkeypatch.setattr(
        main, "Deforestation",
        SimpleNamespace(objects=lambda **k: SimpleNamespace(first=lambda: None)),
        raising=False,
    )

    assert main.run_save_to_db("Paso 6", False, _save_params(dm, available_periods=[])) is False
    assert "Sin deforestation_id para período" in capsys.readouterr().out


def test_run_save_to_db_looks_up_layer_in_database(tmp_path, monkeypatch, capsys):
    dm = SaveDM(tmp_path)
    _write_total_and_enterprise(dm)
    monkeypatch.setattr(
        main, "Deforestation",
        SimpleNamespace(
            objects=lambda **k: SimpleNamespace(first=lambda: SimpleNamespace(id="defo_bd"))
        ),
        raising=False,
    )

    assert main.run_save_to_db("Paso 6", False, _save_params(dm, available_periods=[])) is True
    assert "Encontrado en BD" in capsys.readouterr().out


def test_run_save_to_db_reports_layer_lookup_errors(tmp_path, monkeypatch, capsys):
    dm = SaveDM(tmp_path)

    def explota(**kwargs):
        raise RuntimeError("mongo caído")

    monkeypatch.setattr(main, "Deforestation", SimpleNamespace(objects=explota), raising=False)

    main.run_save_to_db("Paso 6", False, _save_params(dm, available_periods=[]))

    assert "Error buscando capa" in capsys.readouterr().out


def test_run_save_to_db_skips_periods_without_analysis(tmp_path, capsys):
    dm = SaveDM(tmp_path)
    dm.analysis_error = "no se pudo crear"

    assert main.run_save_to_db("Paso 6", False, _save_params(dm)) is False
    assert "Error creando Analysis" in capsys.readouterr().out


def test_run_save_to_db_reports_missing_csvs(tmp_path, capsys):
    dm = SaveDM(tmp_path)

    assert main.run_save_to_db("Paso 6", False, _save_params(dm)) is False
    salida = capsys.readouterr().out
    assert "No existe CSV de total_risk" in salida
    assert "No existe CSV de alertas empresa" in salida


def test_run_save_to_db_reports_read_errors(tmp_path, monkeypatch, capsys):
    dm = SaveDM(tmp_path)
    _write_total_and_enterprise(dm)
    monkeypatch.setattr(
        main.__dict__.setdefault("pd", pd), "read_csv",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("csv roto")),
        raising=False,
    )

    assert main.run_save_to_db("Paso 6", False, _save_params(dm)) is False
    salida = capsys.readouterr().out
    assert "Error leyendo/guardando FarmRisk" in salida


def test_run_save_to_db_prints_granular_errors(tmp_path, capsys):
    dm = SaveDM(tmp_path)
    _write_total_and_enterprise(dm)
    dm.farm_result = (0, 5, [f"error {i}" for i in range(5)])
    dm.enterprise_result = (0, 1, ["fallo empresa"])

    main.run_save_to_db("Paso 6", False, _save_params(dm))

    salida = capsys.readouterr().out
    assert "y 2 errores más" in salida
    assert "fallo empresa" in salida


def test_run_save_to_db_prints_bulk_errors(tmp_path, capsys):
    dm = SaveDM(tmp_path)
    _write_total_and_enterprise(dm)
    dm.farm_result = (0, 3, [{"error": "duplicado", "count": 3}, "individual"])

    main.run_save_to_db("Paso 6", False, _save_params(dm, bulk_insert=True))

    salida = capsys.readouterr().out
    assert "Chunk error: duplicado" in salida


def test_run_save_to_db_summarizes_many_individual_bulk_errors(tmp_path, capsys):
    dm = SaveDM(tmp_path)
    _write_total_and_enterprise(dm)
    dm.farm_result = (0, 9, [f"individual {i}" for i in range(9)])

    main.run_save_to_db("Paso 6", False, _save_params(dm, bulk_insert=True))

    assert "9 errores individuales" in capsys.readouterr().out


# ===================== run_subprocess_script =====================


def test_run_subprocess_script_reports_success(tmp_path, monkeypatch, capsys):
    script = tmp_path / "s.py"
    script.write_text("print('ok')", encoding="utf-8")

    import subprocess

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )

    assert main.run_subprocess_script(script, "Paso X", False, {"VAR": "1"}) is True
    assert "finalizado correctamente" in capsys.readouterr().out


def test_run_subprocess_script_reports_failure(tmp_path, monkeypatch):
    script = tmp_path / "s.py"
    script.write_text("raise SystemExit(1)", encoding="utf-8")

    import subprocess

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=1))

    assert main.run_subprocess_script(script, "Paso X", False) is False
    assert main.run_subprocess_script(script, "Paso X", True) is True


# ===================== main() =====================


class MainDataManager:
    """DataManager de prueba con la superficie que consume main()."""

    instances = []

    def __init__(self, workspace_dir, geoserver_url, geoserver_user, geoserver_pass):
        self.workspace_dir = Path(workspace_dir) / "alertas"
        self.geojsons_dir = self.workspace_dir / "farms" / "geojsons"
        self.geojsons_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.workspace_dir / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.geoserver_url = geoserver_url
        self.calls = []
        # Comportamiento configurable desde cada prueba.
        self.metadata = [{"sitcode": "A1", "mongo_id": "m1", "adm3_id": "adm1"}]
        self.db_error = None
        self.geojsons_available = 1
        self.geometry_stats = {"loaded": 1, "failed": 0, "cache_size_mb": 0.5}
        MainDataManager.instances.append(self)

    def load_farms_metadata(self, limit=None, offline_mode=False, value_chain=None, refresh_data=False):
        self.calls.append(("load_farms_metadata", limit, value_chain, refresh_data))
        metadata = self.metadata
        if limit and metadata:
            metadata = metadata[:limit]
        return (metadata, self.db_error)

    def prepare_geojsons(self, farms_metadata, force_download=False):
        self.calls.append(("prepare_geojsons", force_download))
        return self.geojsons_available

    def load_geometries_to_cache(self, farms_metadata):
        return self.geometry_stats

    def build_spatial_index(self):
        return {"indexed": 1, "build_time": 0.1}

    def get_results_dir(self, stage, source=None, deforestation_type=None, period=None):
        ruta = self.results_dir / (source or "") / (deforestation_type or "") / stage
        ruta.mkdir(parents=True, exist_ok=True)
        return ruta

    def get_mongo_id_map_df(self):
        return pd.DataFrame()


@pytest.fixture
def main_env(tmp_path, monkeypatch):
    """Aisla main(): enums, logging, base de datos y etapas quedan simulados."""
    from enum import Enum

    MainDataManager.instances = []

    DeforestationSource = Enum("DeforestationSource", {"SMBYC": "smbyc"})
    # Se anade un tipo extra para poder ejercitar la validacion manual que hay
    # despues de argparse.
    DeforestationType = Enum(
        "DeforestationType",
        {"ANNUAL": "annual", "CUMULATIVE": "cumulative", "NAD": "nad", "ATD": "atd", "OTRO": "otro"},
    )
    ValueChain = Enum("ValueChain", {"LIVESTOCK": "livestock", "CACAO": "cacao"})

    monkeypatch.setattr(main, "HAS_ORM", True)
    monkeypatch.setattr(main, "HAS_DATA_MANAGER", True)
    monkeypatch.setattr(main, "DeforestationSource", DeforestationSource, raising=False)
    monkeypatch.setattr(main, "DeforestationType", DeforestationType, raising=False)
    monkeypatch.setattr(main, "ValueChain", ValueChain, raising=False)
    monkeypatch.setattr(main, "DataManager", MainDataManager, raising=False)

    monkeypatch.setattr(main, "setup_logging", lambda **kwargs: None)
    logger_falso = SimpleNamespace(
        filepath=tmp_path / "errores.csv", print_summary=lambda: None
    )
    monkeypatch.setattr(main, "init_error_logger", lambda *a, **k: logger_falso)
    monkeypatch.setattr(main, "get_error_logger", lambda: logger_falso)
    monkeypatch.setattr(main, "connect_mongodb", lambda: True)

    monkeypatch.setitem(main.config, "WORKSPACE_DIR", str(tmp_path / "data"))
    monkeypatch.setitem(main.config, "GEOSERVER_URL", "http://geo.example")

    llamadas = []

    def registrar(nombre, resultado=True):
        def _fn(*args, **kwargs):
            llamadas.append(nombre)
            return resultado

        return _fn

    estado = {
        "periods": [
            (datetime(2024, 1, 1), datetime(2024, 3, 31), "smbyc_deforestation_nad_202401", "defo1")
        ],
        "llamadas": llamadas,
        "logger": logger_falso,
        "tmp_path": tmp_path,
    }

    monkeypatch.setattr(main, "query_available_periods", lambda s, pt: estado["periods"])
    # La validación de prerequisitos se prueba aparte; aquí se deja pasar para
    # poder ejercitar la orquestación sin montar todos los insumos en disco.
    monkeypatch.setattr(main, "validate_prerequisites", lambda **kwargs: (True, ""))
    for nombre in (
        "run_direct_alerts",
        "run_indirect_alerts",
        "run_spatial_metrics",
        "run_enterprise_alerts",
        "run_total_risk",
        "run_save_to_db",
    ):
        monkeypatch.setattr(main, nombre, registrar(nombre))

    monkeypatch.setattr(
        main, "calculate_adm3_risk_batch",
        lambda **kwargs: (llamadas.append("adm3"), (1, 0, ["a.csv"]))[1],
    )
    monkeypatch.setattr(
        main, "save_adm3_risk_from_csv_batch",
        lambda **kwargs: (llamadas.append("save_adm3"), (1, 0, 10, 0))[1],
    )
    monkeypatch.setattr(
        main, "calculate_and_save_supplier_risk",
        lambda **kwargs: (llamadas.append("supplier"), {"success": True})[1],
    )
    return estado


def _run_main(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    with pytest.raises(SystemExit) as exc:
        main.main()
    return exc.value.code


def test_main_runs_default_pipeline(main_env, monkeypatch, capsys):
    codigo = _run_main(monkeypatch, "-s", "smbyc", "-pt", "nad", "-y", "2024")

    assert codigo == 0
    assert main_env["llamadas"] == [
        "run_direct_alerts",
        "run_indirect_alerts",
        "run_spatial_metrics",
        "run_total_risk",
        "run_save_to_db",
    ]
    salida = capsys.readouterr().out
    assert "Pipeline completado" in salida
    assert "Ejecutando pipeline COMPLETO" in salida


def test_main_runs_selected_steps(main_env, monkeypatch):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "3")

    assert main_env["llamadas"] == ["run_direct_alerts", "run_spatial_metrics"]


def test_main_skips_standalone_enterprise_when_step_two_runs(main_env, monkeypatch, capsys):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "2", "4")

    assert main_env["llamadas"] == ["run_indirect_alerts"]
    assert "omitido" in capsys.readouterr().out


def test_main_runs_standalone_enterprise_step(main_env, monkeypatch):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "4")

    assert main_env["llamadas"] == ["run_enterprise_alerts"]


def test_main_lists_steps_when_p_has_no_value(main_env, monkeypatch, capsys):
    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p")

    assert codigo == 0
    salida = capsys.readouterr().out
    assert "Pasos disponibles" in salida
    assert "9: Paso 9" in salida


def test_main_rejects_invalid_period_type(main_env, monkeypatch, capsys):
    codigo = _run_main(monkeypatch, "-pt", "otro", "-y", "2024")

    assert codigo == 1
    assert "--period-type debe ser uno de" in capsys.readouterr().out


def test_main_exits_when_requested_periods_are_missing(main_env, monkeypatch, capsys):
    # Hay capas en BD, pero ninguna del anio pedido.
    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "1999")

    assert codigo == 1
    salida = capsys.readouterr().out
    assert "No hay capas de deforestacion disponibles" in salida or "disponibles" in salida
    assert "base de datos" in salida


def test_main_lists_only_first_available_periods(main_env, monkeypatch, capsys):
    main_env["periods"] = [
        (datetime(2020, 1, 1), datetime(2020, 3, 31), f"smbyc_deforestation_nad_20200{i}", f"d{i}")
        for i in range(1, 5)
    ] + [
        (datetime(2021, 1, 1), datetime(2021, 3, 31), f"smbyc_deforestation_nad_20210{i}", f"e{i}")
        for i in range(1, 5)
    ] + [
        (datetime(2022, 1, 1), datetime(2022, 3, 31), f"smbyc_deforestation_nad_20220{i}", f"f{i}")
        for i in range(1, 5)
    ]

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "1999")

    assert codigo == 1
    assert "y 2 m" in capsys.readouterr().out


def test_main_falls_back_to_generated_quarters(main_env, monkeypatch, capsys):
    main_env["periods"] = []

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024")

    assert codigo == 0
    assert "fallback activado" in capsys.readouterr().out


def test_main_falls_back_to_generated_annual_periods(main_env, monkeypatch, capsys):
    main_env["periods"] = []

    codigo = _run_main(monkeypatch, "-pt", "annual", "-y", "2010-2014")

    assert codigo == 0
    # El primer periodo salta 2011, que no tiene datos.
    assert "2010-2012" in capsys.readouterr().out


def test_main_falls_back_to_generated_annual_from_later_year(main_env, monkeypatch, capsys):
    main_env["periods"] = []

    _run_main(monkeypatch, "-pt", "annual", "-y", "2015-2017")

    salida = capsys.readouterr().out
    assert "2015-2016" in salida
    assert "2016-2017" in salida


def test_main_falls_back_to_generated_cumulative_periods(main_env, monkeypatch, capsys):
    main_env["periods"] = []

    _run_main(monkeypatch, "-pt", "cumulative", "-y", "2010-2013")

    salida = capsys.readouterr().out
    assert "2010-2012" in salida
    assert "2010-2013" in salida


def test_main_falls_back_to_cumulative_from_later_year(main_env, monkeypatch, capsys):
    main_env["periods"] = []

    _run_main(monkeypatch, "-pt", "cumulative", "-y", "2015-2017")

    assert "2015-2016" in capsys.readouterr().out


def test_main_falls_back_to_single_quarter_year(main_env, monkeypatch, capsys):
    main_env["periods"] = []

    _run_main(monkeypatch, "-pt", "atd", "-y", "2024")

    assert "202401" in capsys.readouterr().out


def test_main_exits_when_fallback_cannot_generate_periods(main_env, monkeypatch, capsys):
    main_env["periods"] = []

    codigo = _run_main(monkeypatch, "-pt", "annual", "-y", "2024")

    assert codigo == 1
    assert "fallback" in capsys.readouterr().out


def test_main_warns_about_partially_available_ranges(main_env, monkeypatch, capsys):
    main_env["periods"] = [
        (datetime(2017, 1, 1), datetime(2018, 1, 1), "smbyc_deforestation_annual_2017-2018", "d1")
    ]

    _run_main(monkeypatch, "-pt", "annual", "-y", "2017-2020")

    salida = capsys.readouterr().out
    assert "ADVERTENCIA" in salida
    assert "Rango real a procesar" in salida


def test_main_reports_enterprise_argument(main_env, monkeypatch, capsys):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-e", "ACME", "-p", "1")

    assert "ACME" in capsys.readouterr().out


def test_main_reports_bulk_insert_mode(main_env, monkeypatch, capsys):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "--bulk-insert")

    assert "ACTIVADO" in capsys.readouterr().out


def test_main_runs_in_offline_mode(main_env, monkeypatch, capsys):
    dm_dir = main_env["tmp_path"] / "data" / "alertas" / "farms" / "geojsons"
    dm_dir.mkdir(parents=True, exist_ok=True)
    (dm_dir / "A1.geojson").write_text("{}", encoding="utf-8")
    main_env["periods"] = []

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "--offline")

    assert codigo == 0
    salida = capsys.readouterr().out
    assert "OFFLINE" in salida


def test_main_applies_farm_limit(main_env, monkeypatch, capsys):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "--farm-limit", "1")

    assert "TESTING" in capsys.readouterr().out
    assert MainDataManager.instances[0].calls[0] == ("load_farms_metadata", 1, "livestock", False)


def test_main_forwards_refresh_data(main_env, monkeypatch):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "--refresh-data")

    dm = MainDataManager.instances[0]
    assert dm.calls[0][3] is True
    assert ("prepare_geojsons", True) in dm.calls


def test_main_continues_when_database_fails(main_env, monkeypatch, capsys):
    monkeypatch.setattr(
        MainDataManager, "load_farms_metadata",
        lambda self, **kwargs: ([], "sin conexion"),
    )

    codigo = _run_main(
        monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "--continue-on-error"
    )

    assert codigo == 0
    assert "ERROR DE BASE DE DATOS" in capsys.readouterr().out


def test_main_warns_when_no_geojsons_are_available(main_env, monkeypatch, capsys):
    monkeypatch.setattr(MainDataManager, "prepare_geojsons", lambda self, m, force_download=False: 0)

    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1")

    assert "No hay geojsons disponibles para procesar" in capsys.readouterr().out


def test_main_survives_spatial_index_failures(main_env, monkeypatch, capsys):
    monkeypatch.setattr(
        MainDataManager, "build_spatial_index",
        lambda self: (_ for _ in ()).throw(RuntimeError("rtree roto")),
    )

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1")

    assert codigo == 0
    assert "indice espacial" in capsys.readouterr().out.replace("í", "i")


def test_main_skips_geometry_loading_for_steps_that_do_not_need_it(main_env, monkeypatch, capsys):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "5")

    assert "Omitiendo carga de geometr" in capsys.readouterr().out


def test_main_reports_missing_workspace(main_env, monkeypatch, capsys):
    monkeypatch.setitem(main.config, "WORKSPACE_DIR", "")

    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "5", "--continue-on-error")

    assert "WORKSPACE_DIR no configurado" in capsys.readouterr().out


def test_main_reports_data_manager_initialisation_errors(main_env, monkeypatch, capsys):
    def explota(**kwargs):
        raise RuntimeError("workspace invalido")

    monkeypatch.setattr(main, "DataManager", explota)

    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "5", "--continue-on-error")

    assert "Error inicializando DataManager" in capsys.readouterr().out


def test_main_stops_when_prerequisites_fail(main_env, monkeypatch, capsys):
    monkeypatch.setattr(
        main, "validate_prerequisites", lambda **kwargs: (False, "faltan insumos")
    )

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1")

    assert codigo == 1
    assert "Prerequisitos no cumplidos" in capsys.readouterr().out


def test_main_ignores_failed_prerequisites_with_flag(main_env, monkeypatch):
    monkeypatch.setattr(
        main, "validate_prerequisites", lambda **kwargs: (False, "faltan insumos")
    )

    codigo = _run_main(
        monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "--continue-on-error"
    )

    assert codigo == 0


def test_main_stops_pipeline_when_a_stage_fails(main_env, monkeypatch, capsys):
    monkeypatch.setattr(main, "run_direct_alerts", lambda *a, **k: False)

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "2")

    assert codigo == 1
    salida = capsys.readouterr().out
    assert "Pipeline detenido" in salida
    assert "errores" in salida


def test_main_continues_after_stage_failure_with_flag(main_env, monkeypatch):
    monkeypatch.setattr(main, "run_direct_alerts", lambda *a, **k: False)

    codigo = _run_main(
        monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "2", "--continue-on-error"
    )

    assert codigo == 1
    assert "run_indirect_alerts" in main_env["llamadas"]


def test_main_runs_adm3_step(main_env, monkeypatch):
    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "7")

    assert codigo == 0
    assert "adm3" in main_env["llamadas"]


def test_main_adm3_step_requires_metadata(main_env, monkeypatch, capsys):
    monkeypatch.setattr(MainDataManager, "load_farms_metadata", lambda self, **k: ([], None))

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "7")

    assert codigo == 1
    assert "farms_metadata no disponible" in capsys.readouterr().out


def test_main_adm3_step_reports_periods_without_layer_id(main_env, monkeypatch, capsys):
    # Periodos sin mongo_id: no hay con que resolver el Analysis.
    main_env["periods"] = [
        (datetime(2024, 1, 1), datetime(2024, 3, 31), "smbyc_deforestation_nad_202401", None)
    ]

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "7")

    assert codigo == 1
    assert "deforestation_id" in capsys.readouterr().out


def test_main_adm3_step_reports_errors(main_env, monkeypatch, capsys):
    monkeypatch.setattr(
        main, "calculate_adm3_risk_batch",
        lambda **k: (_ for _ in ()).throw(RuntimeError("adm3 roto")),
    )

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "7")

    assert codigo == 1
    assert "Error calculando ADM3" in capsys.readouterr().out


def test_main_runs_save_adm3_step(main_env, monkeypatch):
    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "8")

    assert codigo == 0
    assert "save_adm3" in main_env["llamadas"]


def test_main_save_adm3_step_reports_errors(main_env, monkeypatch, capsys):
    monkeypatch.setattr(
        main, "save_adm3_risk_from_csv_batch",
        lambda **k: (_ for _ in ()).throw(RuntimeError("guardado roto")),
    )

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "8")

    assert codigo == 1
    assert "Error guardando ADM3" in capsys.readouterr().out


def test_main_runs_supplier_step(main_env, monkeypatch):
    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "9", "-e", "ACME")

    assert codigo == 0
    assert "supplier" in main_env["llamadas"]


def test_main_supplier_step_requires_enterprise(main_env, monkeypatch, capsys):
    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "9")

    assert codigo == 1
    assert "empresa" in capsys.readouterr().out


def test_main_supplier_step_reports_errors(main_env, monkeypatch, capsys):
    monkeypatch.setattr(
        main, "calculate_and_save_supplier_risk",
        lambda **k: (_ for _ in ()).throw(RuntimeError("suppliers rotos")),
    )

    codigo = _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "9", "-e", "ACME")

    assert codigo == 1
    assert "Error calculando supplier risk" in capsys.readouterr().out


def test_main_supplier_step_forwards_dry_run(main_env, monkeypatch):
    capturado = {}
    monkeypatch.setattr(
        main, "calculate_and_save_supplier_risk",
        lambda **kwargs: (capturado.update(kwargs), {"success": True})[1],
    )

    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "9", "-e", "ACME", "--dry-run")

    assert capturado["dry_run"] is True


def test_main_accepts_cacao_value_chain(main_env, monkeypatch):
    _run_main(monkeypatch, "-pt", "nad", "-y", "2024", "-p", "1", "-vc", "cacao")

    assert MainDataManager.instances[0].calls[0][2] == "cacao"
