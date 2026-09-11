from pathlib import Path

import pandas as pd
import pytest

import adm3_alert
from orm_doubles import Deforestation, Ref, make_document, oid


VALUE_CHAINS = {"LIVESTOCK": "livestock", "CACAO": "cacao"}


@pytest.fixture
def orm(monkeypatch):
    """Instala dobles del ORM y devuelve las clases para preparar cada escenario."""
    from enum import Enum

    ValueChain = Enum("ValueChain", VALUE_CHAINS)

    Adm3 = make_document("Adm3")
    Adm3Risk = make_document("Adm3Risk")
    Analysis = make_document("Analysis")
    FarmRisk = make_document("FarmRisk")

    monkeypatch.setattr(adm3_alert, "HAS_ORM", True)
    monkeypatch.setattr(adm3_alert, "Adm3", Adm3, raising=False)
    monkeypatch.setattr(adm3_alert, "Adm3Risk", Adm3Risk, raising=False)
    monkeypatch.setattr(adm3_alert, "Analysis", Analysis, raising=False)
    monkeypatch.setattr(adm3_alert, "FarmRisk", FarmRisk, raising=False)
    monkeypatch.setattr(adm3_alert, "ValueChain", ValueChain, raising=False)
    monkeypatch.setattr(adm3_alert, "ObjectId", oid, raising=False)

    return {
        "Adm3": Adm3,
        "Adm3Risk": Adm3Risk,
        "Analysis": Analysis,
        "FarmRisk": FarmRisk,
        "ValueChain": ValueChain,
    }


@pytest.fixture
def no_orm(monkeypatch):
    monkeypatch.setattr(adm3_alert, "HAS_ORM", False)


# ===================== mapeos =====================


def test_build_farm_to_adm3_map_ignores_incomplete_records():
    farms = [
        {"mongo_id": "f1", "adm3_id": "a1"},
        {"mongo_id": "f2", "adm3_id": "a1"},
        {"mongo_id": "f3", "adm3_id": None},
        {"mongo_id": None, "adm3_id": "a2"},
        {},
    ]

    assert adm3_alert.build_farm_to_adm3_map(farms) == {"f1": "a1", "f2": "a1"}


def test_build_adm3_total_farms_counts_by_unit():
    farms = [
        {"mongo_id": "f1", "adm3_id": "a1"},
        {"mongo_id": "f2", "adm3_id": "a1"},
        {"mongo_id": "f3", "adm3_id": "a2"},
        {"mongo_id": "f4"},
    ]

    assert adm3_alert.build_adm3_total_farms(farms) == {"a1": 2, "a2": 1}


# ===================== get_analysis_for_period =====================


def test_get_analysis_for_period_returns_none_without_orm(no_orm):
    assert adm3_alert.get_analysis_for_period("defo1") is None


def test_get_analysis_for_period_finds_matching_analysis(orm):
    orm["Analysis"].reset(
        [
            orm["Analysis"](id="an1", deforestation_id="defo1", value_chain=orm["ValueChain"].LIVESTOCK),
            orm["Analysis"](id="an2", deforestation_id="defo2", value_chain=orm["ValueChain"].LIVESTOCK),
        ]
    )

    assert adm3_alert.get_analysis_for_period("defo1", "livestock") == "an1"


def test_get_analysis_for_period_returns_none_when_absent(orm):
    orm["Analysis"].reset([])

    assert adm3_alert.get_analysis_for_period("defo1", "livestock") is None


def test_get_analysis_for_period_returns_none_for_unknown_value_chain(orm):
    assert adm3_alert.get_analysis_for_period("defo1", "cadena_inexistente") is None


# ===================== get_all_adm3_ids =====================


def test_get_all_adm3_ids_returns_empty_without_orm(no_orm):
    assert adm3_alert.get_all_adm3_ids() == []


def test_get_all_adm3_ids_lists_every_unit(orm):
    orm["Adm3"].reset([orm["Adm3"](id="a1"), orm["Adm3"](id="a2")])

    assert adm3_alert.get_all_adm3_ids() == ["a1", "a2"]


def test_get_all_adm3_ids_returns_empty_on_query_error(orm, monkeypatch):
    class Explota:
        @staticmethod
        def objects():
            raise RuntimeError("mongo caído")

    monkeypatch.setattr(adm3_alert, "Adm3", Explota)

    assert adm3_alert.get_all_adm3_ids() == []


# ===================== calculate_adm3_risk_for_period =====================


def test_calculate_adm3_risk_returns_empty_without_orm(no_orm):
    assert adm3_alert.calculate_adm3_risk_for_period("an1", {}, ["a1"], {}) == []


def test_calculate_adm3_risk_aggregates_hectares_and_counts(orm):
    orm["FarmRisk"].reset(
        [
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("f1"), risk_direct=True,
                risk_input=False, risk_output=False, deforestation=Deforestation(ha=10.5),
            ),
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("f2"), risk_direct=True,
                risk_input=False, risk_output=False, deforestation=Deforestation(ha=4.5),
            ),
            # Riesgo indirecto: cuenta como finca en riesgo pero no suma hectáreas.
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("f3"), risk_direct=False,
                risk_input=True, risk_output=False, deforestation=Deforestation(ha=99.0),
            ),
        ]
    )

    rows = adm3_alert.calculate_adm3_risk_for_period(
        "an1",
        farm_to_adm3={"f1": "a1", "f2": "a1", "f3": "a2"},
        all_adm3_ids=["a1", "a2", "a3"],
        adm3_total_farms={"a1": 5, "a2": 3, "a3": 7},
    )

    por_adm3 = {row["adm3_id"]: row for row in rows}

    assert por_adm3["a1"]["def_ha"] == 15.0
    assert por_adm3["a1"]["farm_amount"] == 2
    assert por_adm3["a1"]["farm_amount_total"] == 5
    assert por_adm3["a1"]["risk_total"] is True

    # a2 tiene riesgo indirecto: cuenta la finca pero no acumula hectáreas.
    assert por_adm3["a2"]["def_ha"] == 0.0
    assert por_adm3["a2"]["farm_amount"] == 1
    assert por_adm3["a2"]["risk_total"] is True

    # a3 no tiene fincas en riesgo, pero igual aparece en el CSV.
    assert por_adm3["a3"] == {
        "adm3_id": "a3",
        "analysis_id": "an1",
        "def_ha": 0.0,
        "risk_total": False,
        "farm_amount": 0,
        "farm_amount_total": 7,
    }


def test_calculate_adm3_risk_deduplicates_repeated_farms(orm):
    orm["FarmRisk"].reset(
        [
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("f1"), risk_direct=True,
                risk_input=False, risk_output=False, deforestation=Deforestation(ha=10.0),
            ),
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("f1"), risk_direct=True,
                risk_input=False, risk_output=False, deforestation=Deforestation(ha=10.0),
            ),
        ]
    )

    rows = adm3_alert.calculate_adm3_risk_for_period(
        "an1", {"f1": "a1"}, ["a1"], {"a1": 1}
    )

    # La misma finca no se cuenta ni se suma dos veces.
    assert rows[0]["farm_amount"] == 1
    assert rows[0]["def_ha"] == 10.0


def test_calculate_adm3_risk_skips_farms_without_id_or_mapping(orm):
    orm["FarmRisk"].reset(
        [
            orm["FarmRisk"](
                analysis_id="an1", farm_id=None, risk_direct=True,
                risk_input=False, risk_output=False, deforestation=Deforestation(ha=5.0),
            ),
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("desconocida"), risk_direct=True,
                risk_input=False, risk_output=False, deforestation=Deforestation(ha=5.0),
            ),
        ]
    )

    rows = adm3_alert.calculate_adm3_risk_for_period("an1", {"f1": "a1"}, ["a1"], {"a1": 1})

    assert rows[0]["farm_amount"] == 0
    assert rows[0]["risk_total"] is False


def test_calculate_adm3_risk_handles_missing_deforestation_block(orm):
    orm["FarmRisk"].reset(
        [
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("f1"), risk_direct=True,
                risk_input=False, risk_output=False, deforestation=None,
            ),
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("f2"), risk_direct=True,
                risk_input=False, risk_output=False, deforestation=Deforestation(ha=None),
            ),
        ]
    )

    rows = adm3_alert.calculate_adm3_risk_for_period(
        "an1", {"f1": "a1", "f2": "a1"}, ["a1"], {"a1": 2}
    )

    assert rows[0]["farm_amount"] == 2
    assert rows[0]["def_ha"] == 0.0


def test_calculate_adm3_risk_returns_empty_on_query_error(orm, monkeypatch):
    monkeypatch.setattr(
        adm3_alert, "ObjectId", lambda v=None: (_ for _ in ()).throw(ValueError("id inválido"))
    )

    assert adm3_alert.calculate_adm3_risk_for_period("malo", {}, ["a1"], {}) == []


# ===================== calculate_adm3_risk_batch =====================


@pytest.fixture
def batch_scenario(orm):
    orm["Adm3"].reset([orm["Adm3"](id="a1"), orm["Adm3"](id="a2")])
    orm["Analysis"].reset(
        [orm["Analysis"](id="an1", deforestation_id="defo1", value_chain=orm["ValueChain"].LIVESTOCK)]
    )
    orm["FarmRisk"].reset(
        [
            orm["FarmRisk"](
                analysis_id="an1", farm_id=Ref("f1"), risk_direct=True,
                risk_input=False, risk_output=False, deforestation=Deforestation(ha=3.0),
            )
        ]
    )
    return orm


def test_calculate_adm3_risk_batch_returns_empty_without_orm(no_orm):
    assert adm3_alert.calculate_adm3_risk_batch([], [], output_dir=None) == (0, 0, [])


def test_calculate_adm3_risk_batch_writes_csv_per_period(batch_scenario, tmp_path):
    farms = [{"mongo_id": "f1", "adm3_id": "a1"}, {"mongo_id": "f2", "adm3_id": "a2"}]

    processed, failed, paths = adm3_alert.calculate_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo1")],
        farms_metadata=farms,
        value_chain="livestock",
        output_dir=tmp_path,
        source="smbyc",
        period_type="nad",
    )

    assert (processed, failed) == (1, 0)
    esperado = tmp_path / "smbyc" / "nad" / "adm3_risk" / "smbyc_adm3_risk_nad_202401.csv"
    assert paths == [str(esperado)]

    df = pd.read_csv(esperado)
    assert list(df.columns) == [
        "adm3_id",
        "analysis_id",
        "def_ha",
        "risk_total",
        "farm_amount",
        "farm_amount_total",
    ]
    assert len(df) == 2


def test_calculate_adm3_risk_batch_extracts_annual_period_code(batch_scenario, tmp_path):
    _, _, paths = adm3_alert.calculate_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_annual_2013-2014", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
        period_type="annual",
    )

    assert paths[0].endswith("smbyc_adm3_risk_annual_2013-2014.csv")


def test_calculate_adm3_risk_batch_falls_back_to_cwd_without_output_dir(
    batch_scenario, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    _, _, paths = adm3_alert.calculate_adm3_risk_batch(
        periods_with_defo_id=[("periodo_raro", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=None,
    )

    assert paths == ["adm3_risk_periodo_raro.csv"]
    assert (tmp_path / "adm3_risk_periodo_raro.csv").exists()


def test_calculate_adm3_risk_batch_counts_period_without_analysis(batch_scenario, tmp_path):
    processed, failed, paths = adm3_alert.calculate_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo_inexistente")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
    )

    assert (processed, failed, paths) == (0, 1, [])


def test_calculate_adm3_risk_batch_counts_period_without_rows(batch_scenario, tmp_path, monkeypatch):
    monkeypatch.setattr(adm3_alert, "calculate_adm3_risk_for_period", lambda *a, **k: [])

    processed, failed, _ = adm3_alert.calculate_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
    )

    assert (processed, failed) == (0, 1)


def test_calculate_adm3_risk_batch_captures_unexpected_errors(batch_scenario, tmp_path, monkeypatch):
    monkeypatch.setattr(
        adm3_alert,
        "calculate_adm3_risk_for_period",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cálculo roto")),
    )

    processed, failed, _ = adm3_alert.calculate_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
    )

    assert (processed, failed) == (0, 1)


def test_calculate_adm3_risk_batch_lists_only_first_paths(batch_scenario, tmp_path, capsys):
    periodos = [(f"smbyc_deforestation_nad_20240{i}", "defo1") for i in range(1, 5)]
    periodos += [("smbyc_deforestation_nad_202305", "defo1"), ("smbyc_deforestation_nad_202306", "defo1")]

    adm3_alert.calculate_adm3_risk_batch(
        periods_with_defo_id=periodos,
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
    )

    assert "y 1 más" in capsys.readouterr().out


# ===================== save_adm3_risk_to_db =====================


def _rows(n=2):
    return [
        {
            "adm3_id": f"a{i}",
            "analysis_id": "an1",
            "def_ha": float(i),
            "risk_total": i == 1,
            "farm_amount": i,
            "farm_amount_total": 10,
        }
        for i in range(1, n + 1)
    ]


def test_save_adm3_risk_to_db_requires_orm(no_orm):
    saved, failed, errors = adm3_alert.save_adm3_risk_to_db(_rows(), "an1")

    assert (saved, failed) == (0, 0)
    assert errors == [{"error": "ORM no disponible"}]


def test_save_adm3_risk_to_db_rejects_invalid_analysis_id(orm, monkeypatch):
    monkeypatch.setattr(
        adm3_alert, "ObjectId", lambda v=None: (_ for _ in ()).throw(ValueError("no es un oid"))
    )

    saved, failed, errors = adm3_alert.save_adm3_risk_to_db(_rows(), "malo")

    assert (saved, failed) == (0, 0)
    assert "analysis_id inválido" in errors[0]["error"]


def test_save_adm3_risk_to_db_inserts_new_documents(orm):
    orm["Adm3Risk"].reset([])

    saved, failed, errors = adm3_alert.save_adm3_risk_to_db(_rows(2), "an1")

    assert (saved, failed, errors) == (2, 0, [])
    assert len(orm["Adm3Risk"]._store) == 2
    guardado = orm["Adm3Risk"]._store[0]
    assert guardado.def_ha == 1.0
    assert guardado.farm_total_amount == 10


def test_save_adm3_risk_to_db_updates_existing_documents(orm):
    existente = orm["Adm3Risk"](
        adm3_id="a1", analysis_id="an1", def_ha=0.0, farm_amount=0,
        risk_total=False, farm_total_amount=0,
    )
    orm["Adm3Risk"].reset([existente])

    saved, failed, _ = adm3_alert.save_adm3_risk_to_db(_rows(1), "an1")

    assert (saved, failed) == (1, 0)
    # Se actualiza en sitio, no se duplica.
    assert len(orm["Adm3Risk"]._store) == 1
    assert existente.def_ha == 1.0
    assert existente.risk_total is True


def test_save_adm3_risk_to_db_skips_rows_without_adm3_id(orm):
    orm["Adm3Risk"].reset([])

    saved, failed, _ = adm3_alert.save_adm3_risk_to_db([{"adm3_id": None}], "an1")

    assert (saved, failed) == (0, 1)


def test_save_adm3_risk_to_db_records_individual_failures(orm):
    orm["Adm3Risk"].reset([])
    orm["Adm3Risk"].save_error = RuntimeError("escritura rechazada")

    saved, failed, errors = adm3_alert.save_adm3_risk_to_db(_rows(2), "an1")

    assert (saved, failed) == (0, 2)
    assert errors[0]["adm3_id"] == "a1"
    assert "escritura rechazada" in errors[0]["error"]


# ===================== save_adm3_risk_to_db_bulk =====================


def test_save_adm3_risk_to_db_bulk_requires_orm(no_orm):
    saved, failed, errors = adm3_alert.save_adm3_risk_to_db_bulk(_rows(), "an1")

    assert errors == [{"error": "ORM no disponible"}]


def test_save_adm3_risk_to_db_bulk_rejects_invalid_analysis_id(orm, monkeypatch):
    monkeypatch.setattr(
        adm3_alert, "ObjectId", lambda v=None: (_ for _ in ()).throw(ValueError("no es un oid"))
    )

    saved, failed, errors = adm3_alert.save_adm3_risk_to_db_bulk(_rows(), "malo")

    assert "analysis_id inválido" in errors[0]["error"]


def test_save_adm3_risk_to_db_bulk_replaces_previous_analysis_rows(orm, capsys):
    previos = [
        orm["Adm3Risk"](adm3_id="viejo1", analysis_id="an1"),
        orm["Adm3Risk"](adm3_id="viejo2", analysis_id="an1"),
        orm["Adm3Risk"](adm3_id="otro", analysis_id="an2"),
    ]
    orm["Adm3Risk"].reset(previos)

    saved, failed, errors = adm3_alert.save_adm3_risk_to_db_bulk(_rows(2), "an1")

    assert (saved, failed, errors) == (2, 0, [])
    assert "Eliminados 2" in capsys.readouterr().out
    # El análisis ajeno sobrevive.
    ids = [d.adm3_id for d in orm["Adm3Risk"]._store]
    assert "otro" in ids
    assert "viejo1" not in ids


def test_save_adm3_risk_to_db_bulk_tolerates_delete_errors(orm, monkeypatch):
    orm["Adm3Risk"].reset([])

    class ObjetosQueFallanAlBorrar:
        def __call__(self, **kwargs):
            class QS:
                def delete(self):
                    raise RuntimeError("no se pudo borrar")

            return QS()

        def insert(self, docs, **kwargs):
            return docs

    monkeypatch.setattr(orm["Adm3Risk"], "objects", ObjetosQueFallanAlBorrar())

    saved, failed, _ = adm3_alert.save_adm3_risk_to_db_bulk(_rows(2), "an1")

    # El borrado falla pero el insert continúa.
    assert saved == 2


def test_save_adm3_risk_to_db_bulk_skips_rows_without_adm3_id(orm):
    orm["Adm3Risk"].reset([])

    saved, failed, _ = adm3_alert.save_adm3_risk_to_db_bulk(
        _rows(1) + [{"adm3_id": None}], "an1"
    )

    assert saved == 1
    # La fila descartada se refleja como fallida en el total.
    assert failed == 1


def test_save_adm3_risk_to_db_bulk_records_document_build_errors(orm, monkeypatch):
    orm["Adm3Risk"].reset([])
    llamadas = {"n": 0}
    real_oid = adm3_alert.ObjectId

    def flaky(value=None):
        llamadas["n"] += 1
        if llamadas["n"] > 1 and value == "a2":
            raise ValueError("oid corrupto")
        return real_oid(value)

    monkeypatch.setattr(adm3_alert, "ObjectId", flaky)

    saved, failed, errors = adm3_alert.save_adm3_risk_to_db_bulk(_rows(2), "an1")

    assert saved == 1
    assert errors[0]["adm3_id"] == "a2"


def test_save_adm3_risk_to_db_bulk_reports_chunk_failures(orm):
    orm["Adm3Risk"].reset([])
    orm["Adm3Risk"].insert_error = RuntimeError("clave duplicada")

    saved, failed, errors = adm3_alert.save_adm3_risk_to_db_bulk(_rows(2), "an1")

    assert saved == 0
    assert failed == 2
    assert errors[0]["chunk"] == 1


def test_save_adm3_risk_to_db_bulk_splits_into_chunks(orm):
    orm["Adm3Risk"].reset([])

    saved, _, _ = adm3_alert.save_adm3_risk_to_db_bulk(_rows(5), "an1", chunk_size=2)

    assert saved == 5


# ===================== calculate_and_save_adm3_risk_batch =====================


def test_calculate_and_save_requires_orm(no_orm):
    assert adm3_alert.calculate_and_save_adm3_risk_batch([], []) == (0, 0, [], 0, 0)


def test_calculate_and_save_writes_csv_and_persists(batch_scenario, tmp_path):
    batch_scenario["Adm3Risk"].reset([])

    csv_ok, csv_fail, paths, db_saved, db_failed = adm3_alert.calculate_and_save_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
        save_to_db=True,
        bulk_insert=True,
    )

    assert (csv_ok, csv_fail, db_failed) == (1, 0, 0)
    assert db_saved == 2
    assert Path(paths[0]).exists()


def test_calculate_and_save_uses_granular_mode(batch_scenario, tmp_path):
    batch_scenario["Adm3Risk"].reset([])

    _, _, _, db_saved, _ = adm3_alert.calculate_and_save_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
        save_to_db=True,
        bulk_insert=False,
    )

    assert db_saved == 2


def test_calculate_and_save_can_skip_database(batch_scenario, tmp_path):
    batch_scenario["Adm3Risk"].reset([])

    csv_ok, _, _, db_saved, db_failed = adm3_alert.calculate_and_save_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
        save_to_db=False,
    )

    assert csv_ok == 1
    assert (db_saved, db_failed) == (0, 0)
    assert batch_scenario["Adm3Risk"]._store == []


def test_calculate_and_save_skips_periods_without_analysis(batch_scenario, tmp_path):
    csv_ok, csv_fail, _, _, _ = adm3_alert.calculate_and_save_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "inexistente")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
    )

    assert (csv_ok, csv_fail) == (0, 1)


def test_calculate_and_save_skips_periods_without_rows(batch_scenario, tmp_path, monkeypatch):
    monkeypatch.setattr(adm3_alert, "calculate_adm3_risk_for_period", lambda *a, **k: [])

    csv_ok, csv_fail, _, _, _ = adm3_alert.calculate_and_save_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
    )

    assert (csv_ok, csv_fail) == (0, 1)


def test_calculate_and_save_captures_errors(batch_scenario, tmp_path, monkeypatch):
    monkeypatch.setattr(
        adm3_alert,
        "calculate_adm3_risk_for_period",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("falló")),
    )

    csv_ok, csv_fail, _, _, _ = adm3_alert.calculate_and_save_adm3_risk_batch(
        periods_with_defo_id=[("smbyc_deforestation_nad_202401", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
    )

    assert (csv_ok, csv_fail) == (0, 1)


def test_calculate_and_save_without_output_dir_uses_cwd(batch_scenario, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    batch_scenario["Adm3Risk"].reset([])

    _, _, paths, _, _ = adm3_alert.calculate_and_save_adm3_risk_batch(
        periods_with_defo_id=[("periodo_raro", "defo1")],
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=None,
    )

    assert paths == ["adm3_risk_periodo_raro.csv"]


def test_calculate_and_save_truncates_file_listing(batch_scenario, tmp_path, capsys):
    batch_scenario["Adm3Risk"].reset([])
    periodos = [(f"smbyc_deforestation_nad_2024{i:02d}", "defo1") for i in range(1, 8)]

    adm3_alert.calculate_and_save_adm3_risk_batch(
        periods_with_defo_id=periodos,
        farms_metadata=[{"mongo_id": "f1", "adm3_id": "a1"}],
        output_dir=tmp_path,
    )

    assert "más" in capsys.readouterr().out


# ===================== save_adm3_risk_from_csv_batch =====================


def _write_adm3_csv(results_dir, period, source="smbyc", period_type="nad", analysis_id="an1"):
    path = results_dir / source / period_type / "adm3_risk"
    path.mkdir(parents=True, exist_ok=True)
    csv_path = path / f"{source}_adm3_risk_{period_type}_{period}.csv"
    pd.DataFrame(
        {
            "adm3_id": ["a1", "a2"],
            "analysis_id": [analysis_id, analysis_id],
            "def_ha": [1.0, 0.0],
            "risk_total": [True, False],
            "farm_amount": [1, 0],
            "farm_amount_total": [5, 3],
        }
    ).to_csv(csv_path, index=False)
    return csv_path


def test_save_from_csv_requires_orm(no_orm):
    assert adm3_alert.save_adm3_risk_from_csv_batch(["202401"]) == (0, 0, 0, 0)


def test_save_from_csv_requires_results_dir(orm):
    assert adm3_alert.save_adm3_risk_from_csv_batch(["202401"], results_dir=None) == (0, 0, 0, 0)


def test_save_from_csv_reads_analysis_id_from_file(orm, tmp_path):
    orm["Adm3Risk"].reset([])
    _write_adm3_csv(tmp_path, "202401")

    found, missing, saved, failed = adm3_alert.save_adm3_risk_from_csv_batch(
        periods=["202401"], results_dir=tmp_path, bulk_insert=True
    )

    assert (found, missing, saved, failed) == (1, 0, 2, 0)


def test_save_from_csv_counts_missing_files(orm, tmp_path):
    found, missing, saved, failed = adm3_alert.save_adm3_risk_from_csv_batch(
        periods=["202401"], results_dir=tmp_path
    )

    assert (found, missing, saved, failed) == (0, 1, 0, 0)


def test_save_from_csv_skips_empty_files(orm, tmp_path):
    path = tmp_path / "smbyc" / "nad" / "adm3_risk"
    path.mkdir(parents=True)
    pd.DataFrame(columns=["adm3_id", "analysis_id"]).to_csv(
        path / "smbyc_adm3_risk_nad_202401.csv", index=False
    )

    found, missing, saved, _ = adm3_alert.save_adm3_risk_from_csv_batch(
        periods=["202401"], results_dir=tmp_path
    )

    assert (found, saved) == (1, 0)


def test_save_from_csv_resolves_analysis_from_deforestation_map(orm, tmp_path):
    orm["Adm3Risk"].reset([])
    orm["Analysis"].reset(
        [orm["Analysis"](id="an9", deforestation_id="defo1", value_chain=orm["ValueChain"].LIVESTOCK)]
    )
    # CSV sin analysis_id: hay que resolverlo por deforestation_id.
    path = tmp_path / "smbyc" / "nad" / "adm3_risk"
    path.mkdir(parents=True)
    pd.DataFrame({"adm3_id": ["a1"], "def_ha": [1.0], "farm_amount": [1]}).to_csv(
        path / "smbyc_adm3_risk_nad_202401.csv", index=False
    )

    found, missing, saved, _ = adm3_alert.save_adm3_risk_from_csv_batch(
        periods=["202401"],
        results_dir=tmp_path,
        period_to_defo_id={"202401": "defo1"},
    )

    assert (found, missing, saved) == (1, 0, 1)


def test_save_from_csv_reports_period_without_deforestation_id(orm, tmp_path):
    path = tmp_path / "smbyc" / "nad" / "adm3_risk"
    path.mkdir(parents=True)
    pd.DataFrame({"adm3_id": ["a1"]}).to_csv(path / "smbyc_adm3_risk_nad_202401.csv", index=False)

    found, missing, saved, _ = adm3_alert.save_adm3_risk_from_csv_batch(
        periods=["202401"], results_dir=tmp_path, period_to_defo_id={}
    )

    assert (found, missing, saved) == (1, 1, 0)


def test_save_from_csv_reports_period_without_analysis(orm, tmp_path):
    orm["Analysis"].reset([])
    path = tmp_path / "smbyc" / "nad" / "adm3_risk"
    path.mkdir(parents=True)
    pd.DataFrame({"adm3_id": ["a1"]}).to_csv(path / "smbyc_adm3_risk_nad_202401.csv", index=False)

    found, missing, saved, _ = adm3_alert.save_adm3_risk_from_csv_batch(
        periods=["202401"], results_dir=tmp_path, period_to_defo_id={"202401": "defo_x"}
    )

    assert (found, missing, saved) == (1, 1, 0)


def test_save_from_csv_uses_granular_mode(orm, tmp_path):
    orm["Adm3Risk"].reset([])
    _write_adm3_csv(tmp_path, "202401")

    _, _, saved, _ = adm3_alert.save_adm3_risk_from_csv_batch(
        periods=["202401"], results_dir=tmp_path, bulk_insert=False
    )

    assert saved == 2


def test_save_from_csv_captures_read_errors(orm, tmp_path, monkeypatch):
    _write_adm3_csv(tmp_path, "202401")
    monkeypatch.setattr(
        adm3_alert.pd, "read_csv", lambda *a, **k: (_ for _ in ()).throw(ValueError("csv roto"))
    )

    found, missing, saved, _ = adm3_alert.save_adm3_risk_from_csv_batch(
        periods=["202401"], results_dir=tmp_path
    )

    assert (found, missing, saved) == (1, 1, 0)
