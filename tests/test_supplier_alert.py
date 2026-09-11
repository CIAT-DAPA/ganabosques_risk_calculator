import json
from enum import Enum
from pathlib import Path

import pandas as pd
import pytest

import supplier_alert
from orm_doubles import Ref, make_document, oid


ValueChain = Enum("ValueChain", {"LIVESTOCK": "livestock", "CACAO": "cacao"})
Label = Enum("Label", {"HIGH": "high", "LOW": "low"})


class _EnterpriseQuerySet:
    """QuerySet de Enterprise con los filtros propios de mongoengine que usa el módulo."""

    def __init__(self, items):
        self._items = list(items)

    def first(self):
        return self._items[0] if self._items else None

    def count(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


class FakeEnterpriseManager:
    def __init__(self, enterprises):
        self.enterprises = list(enterprises)

    def __call__(self, **filters):
        vc = filters.pop("value_chain", None)
        pool = [e for e in self.enterprises if vc is None or e.value_chain == vc]

        if "ext_id__ext_code" in filters:
            code = filters["ext_id__ext_code"]
            pool = [
                e for e in pool if any(x.ext_code == code for x in (e.ext_id or []))
            ]
        elif "name__iexact" in filters:
            valor = filters["name__iexact"].lower()
            pool = [e for e in pool if e.name.lower() == valor]
        elif "name__icontains" in filters:
            valor = filters["name__icontains"].lower()
            pool = [e for e in pool if valor in e.name.lower()]

        return _EnterpriseQuerySet(pool)


class ExtId:
    def __init__(self, ext_code):
        self.ext_code = ext_code


class FakeEnterprise:
    def __init__(self, id, name, ext_id=None, type_enterprise="SLAUGHTERHOUSE", value_chain=ValueChain.LIVESTOCK):
        self.id = id
        self.name = name
        self.ext_id = ext_id or []
        self.type_enterprise = type_enterprise
        self.value_chain = value_chain


@pytest.fixture
def orm(monkeypatch):
    Suppliers = make_document("Suppliers")
    Analysis = make_document("Analysis")
    Deforestation = make_document("Deforestation")
    FarmRisk = make_document("FarmRisk")
    EnterpriseRisk = make_document("EnterpriseRisk")

    class EnterpriseHolder:
        objects = FakeEnterpriseManager([])

    monkeypatch.setattr(supplier_alert, "HAS_ORM", True)
    monkeypatch.setattr(supplier_alert, "Enterprise", EnterpriseHolder, raising=False)
    monkeypatch.setattr(supplier_alert, "Suppliers", Suppliers, raising=False)
    monkeypatch.setattr(supplier_alert, "Analysis", Analysis, raising=False)
    monkeypatch.setattr(supplier_alert, "Deforestation", Deforestation, raising=False)
    monkeypatch.setattr(supplier_alert, "FarmRisk", FarmRisk, raising=False)
    monkeypatch.setattr(supplier_alert, "EnterpriseRisk", EnterpriseRisk, raising=False)
    monkeypatch.setattr(supplier_alert, "ValueChain", ValueChain, raising=False)
    monkeypatch.setattr(supplier_alert, "Label", Label, raising=False)
    monkeypatch.setattr(supplier_alert, "ObjectId", oid, raising=False)

    return {
        "Enterprise": EnterpriseHolder,
        "Suppliers": Suppliers,
        "Analysis": Analysis,
        "Deforestation": Deforestation,
        "FarmRisk": FarmRisk,
        "EnterpriseRisk": EnterpriseRisk,
    }


@pytest.fixture
def no_orm(monkeypatch):
    monkeypatch.setattr(supplier_alert, "HAS_ORM", False)


# ===================== find_enterprise =====================


def test_find_enterprise_returns_none_without_orm(no_orm):
    assert supplier_alert.find_enterprise("Acme") is None


@pytest.mark.parametrize("value", ["", "   ", None])
def test_find_enterprise_rejects_blank_input(orm, value):
    assert supplier_alert.find_enterprise(value) is None


def test_find_enterprise_matches_by_external_code(orm):
    acme = FakeEnterprise("e1", "Acme S.A.", ext_id=[ExtId("COD-123")])
    orm["Enterprise"].objects = FakeEnterpriseManager([acme])

    assert supplier_alert.find_enterprise("COD-123") is acme


def test_find_enterprise_matches_by_exact_name_case_insensitively(orm):
    acme = FakeEnterprise("e1", "Acme S.A.")
    orm["Enterprise"].objects = FakeEnterpriseManager([acme])

    assert supplier_alert.find_enterprise("acme s.a.") is acme


def test_find_enterprise_matches_by_unique_partial_name(orm):
    acme = FakeEnterprise("e1", "Frigorífico Acme del Norte")
    orm["Enterprise"].objects = FakeEnterpriseManager([acme])

    assert supplier_alert.find_enterprise("Acme") is acme


def test_find_enterprise_refuses_ambiguous_partial_matches(orm, capsys):
    orm["Enterprise"].objects = FakeEnterpriseManager(
        [
            FakeEnterprise("e1", "Acme Norte", ext_id=[ExtId("C1")]),
            FakeEnterprise("e2", "Acme Sur"),
        ]
    )

    assert supplier_alert.find_enterprise("Acme") is None
    salida = capsys.readouterr().out
    assert "Se encontraron 2 empresas" in salida
    assert "más específico" in salida


def test_find_enterprise_truncates_long_ambiguous_lists(orm, capsys):
    orm["Enterprise"].objects = FakeEnterpriseManager(
        [FakeEnterprise(f"e{i}", f"Acme {i}") for i in range(12)]
    )

    assert supplier_alert.find_enterprise("Acme") is None
    assert "y 2 más" in capsys.readouterr().out


def test_find_enterprise_returns_none_when_nothing_matches(orm):
    orm["Enterprise"].objects = FakeEnterpriseManager([FakeEnterprise("e1", "Otra")])

    assert supplier_alert.find_enterprise("Acme") is None


def test_find_enterprise_filters_by_value_chain(orm):
    ganado = FakeEnterprise("e1", "Acme", value_chain=ValueChain.LIVESTOCK)
    cacao = FakeEnterprise("e2", "Acme", value_chain=ValueChain.CACAO)
    orm["Enterprise"].objects = FakeEnterpriseManager([ganado, cacao])

    assert supplier_alert.find_enterprise("Acme", value_chain="cacao") is cacao


def test_find_enterprise_ignores_invalid_value_chain(orm, capsys):
    acme = FakeEnterprise("e1", "Acme")
    orm["Enterprise"].objects = FakeEnterpriseManager([acme])

    assert supplier_alert.find_enterprise("Acme", value_chain="inexistente") is acme
    assert "no válido" in capsys.readouterr().out


# ===================== load_suppliers_for_enterprise =====================


def test_load_suppliers_returns_empty_without_orm(no_orm):
    assert supplier_alert.load_suppliers_for_enterprise("e1") == []


def test_load_suppliers_reads_from_database(orm, capsys):
    orm["Suppliers"].reset(
        [
            orm["Suppliers"](enterprise_id="e1", farm_id=Ref("f1"), years=[2017, 2018]),
            orm["Suppliers"](enterprise_id="e1", farm_id=Ref("f2"), years=[2019]),
            orm["Suppliers"](enterprise_id="e2", farm_id=Ref("f3"), years=[2020]),
        ]
    )

    result = supplier_alert.load_suppliers_for_enterprise("e1")

    assert result == [
        {"farm_id": "f1", "years": [2017, 2018]},
        {"farm_id": "f2", "years": [2019]},
    ]
    salida = capsys.readouterr().out
    assert "Años cubiertos: 2017-2019" in salida
    assert "Fincas únicas: 2" in salida


def test_load_suppliers_handles_plain_farm_ids_and_empty_years(orm):
    orm["Suppliers"].reset(
        [orm["Suppliers"](enterprise_id="e1", farm_id="f1", years=None)]
    )

    assert supplier_alert.load_suppliers_for_enterprise("e1") == [
        {"farm_id": "f1", "years": []}
    ]


def test_load_suppliers_writes_and_reuses_cache(orm, tmp_path, capsys):
    orm["Suppliers"].reset(
        [orm["Suppliers"](enterprise_id="e1", farm_id=Ref("f1"), years=[2017])]
    )

    primero = supplier_alert.load_suppliers_for_enterprise("e1", cache_dir=tmp_path)
    cache_file = tmp_path / "suppliers_e1.json"
    assert cache_file.exists()
    assert json.loads(cache_file.read_text(encoding="utf-8")) == primero

    # En la segunda llamada no debe tocar la BD.
    orm["Suppliers"].reset([])
    segundo = supplier_alert.load_suppliers_for_enterprise("e1", cache_dir=tmp_path)

    assert segundo == primero
    assert "desde caché" in capsys.readouterr().out


def test_load_suppliers_can_force_reload(orm, tmp_path):
    cache_file = tmp_path / "suppliers_e1.json"
    cache_file.write_text(json.dumps([{"farm_id": "viejo", "years": [1999]}]), encoding="utf-8")
    orm["Suppliers"].reset(
        [orm["Suppliers"](enterprise_id="e1", farm_id=Ref("f1"), years=[2017])]
    )

    result = supplier_alert.load_suppliers_for_enterprise(
        "e1", cache_dir=tmp_path, force_reload=True
    )

    assert result == [{"farm_id": "f1", "years": [2017]}]


def test_load_suppliers_falls_back_to_database_on_corrupt_cache(orm, tmp_path, capsys):
    (tmp_path / "suppliers_e1.json").write_text("{ esto no es json", encoding="utf-8")
    orm["Suppliers"].reset(
        [orm["Suppliers"](enterprise_id="e1", farm_id=Ref("f1"), years=[2017])]
    )

    result = supplier_alert.load_suppliers_for_enterprise("e1", cache_dir=tmp_path)

    assert result == [{"farm_id": "f1", "years": [2017]}]
    assert "Error leyendo caché" in capsys.readouterr().out


def test_load_suppliers_tolerates_cache_write_errors(orm, tmp_path, monkeypatch):
    orm["Suppliers"].reset(
        [orm["Suppliers"](enterprise_id="e1", farm_id=Ref("f1"), years=[2017])]
    )
    monkeypatch.setattr(
        supplier_alert.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disco lleno"))
    )

    assert supplier_alert.load_suppliers_for_enterprise("e1", cache_dir=tmp_path) == [
        {"farm_id": "f1", "years": [2017]}
    ]


def test_load_suppliers_returns_empty_on_query_error(orm, monkeypatch):
    monkeypatch.setattr(
        supplier_alert, "ObjectId", lambda v=None: (_ for _ in ()).throw(ValueError("oid malo"))
    )

    assert supplier_alert.load_suppliers_for_enterprise("e1") == []


# ===================== períodos =====================


def test_get_years_for_period_covers_every_type():
    assert supplier_alert.get_years_for_period("2010-2012", "cumulative") == {2010, 2011, 2012}
    assert supplier_alert.get_years_for_period("201701", "nad") == {2017}
    assert supplier_alert.get_years_for_period("202304", "atd") == {2023}
    assert supplier_alert.get_years_for_period("2017-2018", "annual") == {2017, 2018}
    assert supplier_alert.get_years_for_period("2017", "annual") == {2017}
    # Un valor no numérico sin guion cae en el `else`, que sí atrapa el ValueError.
    assert supplier_alert.get_years_for_period("basura", "annual") == set()


def test_get_years_for_period_propagates_error_for_malformed_ranges():
    # OJO: las ramas con guion no capturan el ValueError, así que un rango mal
    # formado revienta en vez de devolver un conjunto vacío. Se deja fijado para
    # que el día que se corrija el fallo salte esta prueba.
    with pytest.raises(ValueError):
        supplier_alert.get_years_for_period("no-es-un-año", "annual")


def test_expand_year_to_quarters():
    assert supplier_alert.expand_year_to_quarters(2024) == [
        "202401",
        "202402",
        "202403",
        "202404",
    ]


def test_get_suppliers_for_period_returns_all_farms_of_valid_period():
    suppliers = [
        {"farm_id": "f1", "years": [2017]},
        # El filtro por año está deshabilitado a propósito en el código, así que
        # también se incluyen suppliers de otros años.
        {"farm_id": "f2", "years": [1999]},
    ]

    assert supplier_alert.get_suppliers_for_period(suppliers, "2017", "annual") == ["f1", "f2"]


def test_get_suppliers_for_period_returns_empty_for_unparseable_period():
    assert supplier_alert.get_suppliers_for_period([{"farm_id": "f1"}], "xx", "annual") == []


# ===================== get_analysis_for_period =====================


def test_get_analysis_for_period_returns_none_without_orm(no_orm):
    assert supplier_alert.get_analysis_for_period("2017", "annual", "smbyc", "livestock") is None


def test_get_analysis_for_period_resolves_via_deforestation_layer(orm):
    orm["Deforestation"].reset(
        [orm["Deforestation"](id="defo1", name="smbyc_deforestation_annual_2017")]
    )
    orm["Analysis"].reset(
        [orm["Analysis"](id="an1", deforestation_id="defo1", value_chain="livestock")]
    )

    assert (
        supplier_alert.get_analysis_for_period("2017", "annual", "smbyc", "LIVESTOCK") == "an1"
    )


def test_get_analysis_for_period_returns_none_without_layer(orm):
    orm["Deforestation"].reset([])

    assert supplier_alert.get_analysis_for_period("2017", "annual", "smbyc", "livestock") is None


def test_get_analysis_for_period_returns_none_without_analysis(orm):
    orm["Deforestation"].reset(
        [orm["Deforestation"](id="defo1", name="smbyc_deforestation_annual_2017")]
    )
    orm["Analysis"].reset([])

    assert supplier_alert.get_analysis_for_period("2017", "annual", "smbyc", "livestock") is None


def test_get_analysis_for_period_returns_none_on_error(orm, monkeypatch):
    class Explota:
        @staticmethod
        def objects(**kwargs):
            raise RuntimeError("mongo caído")

    monkeypatch.setattr(supplier_alert, "Deforestation", Explota)

    assert supplier_alert.get_analysis_for_period("2017", "annual", "smbyc", "livestock") is None


# ===================== find_farm_risks_for_farms =====================


def test_find_farm_risks_returns_empty_without_orm(no_orm):
    assert supplier_alert.find_farm_risks_for_farms(["f1"], "an1") == []


def test_find_farm_risks_returns_empty_for_empty_farm_list(orm):
    assert supplier_alert.find_farm_risks_for_farms([], "an1") == []


def test_find_farm_risks_filters_by_farm_and_analysis(orm):
    esperado = orm["FarmRisk"](id="fr1", farm_id=Ref("f1"), analysis_id="an1")
    orm["FarmRisk"].reset(
        [
            esperado,
            orm["FarmRisk"](id="fr2", farm_id=Ref("f9"), analysis_id="an1"),
            orm["FarmRisk"](id="fr3", farm_id=Ref("f1"), analysis_id="an2"),
        ]
    )

    result = supplier_alert.find_farm_risks_for_farms(["f1"], "an1")

    assert [fr.id for fr in result] == ["fr1"]


def test_find_farm_risks_returns_empty_on_error(orm, monkeypatch):
    monkeypatch.setattr(
        supplier_alert, "ObjectId", lambda v=None: (_ for _ in ()).throw(ValueError("oid malo"))
    )

    assert supplier_alert.find_farm_risks_for_farms(["f1"], "an1") == []


# ===================== calculate_and_save_supplier_risk =====================


@pytest.fixture
def escenario(orm, monkeypatch):
    """Empresa con un supplier, un analysis y un FarmRisk con riesgo directo."""
    acme = FakeEnterprise("e1", "Acme")
    orm["Enterprise"].objects = FakeEnterpriseManager([acme])
    orm["Suppliers"].reset(
        [orm["Suppliers"](enterprise_id="e1", farm_id=Ref("f1"), years=[2017])]
    )
    orm["Deforestation"].reset(
        [orm["Deforestation"](id="defo1", name="smbyc_deforestation_annual_2017")]
    )
    orm["Analysis"].reset(
        [orm["Analysis"](id="an1", deforestation_id="defo1", value_chain="livestock")]
    )
    orm["FarmRisk"].reset(
        [
            orm["FarmRisk"](
                id="fr1", farm_id=Ref("f1"), analysis_id="an1",
                risk_direct=True, risk_input=False, risk_output=False,
            )
        ]
    )
    orm["EnterpriseRisk"].reset([])
    monkeypatch.setattr(
        supplier_alert, "pkg_supplier_risk", lambda **kwargs: pd.DataFrame([{"id": "f1"}])
    )
    orm["enterprise"] = acme
    return orm


def _run(**kwargs):
    defaults = dict(
        enterprise_name_or_code="Acme",
        periods=["2017"],
        period_type="annual",
    )
    defaults.update(kwargs)
    return supplier_alert.calculate_and_save_supplier_risk(**defaults)


def test_calculate_supplier_risk_requires_orm(no_orm):
    assert _run() == {"success": False, "error": "ganabosques_orm no disponible"}


def test_calculate_supplier_risk_fails_when_enterprise_is_missing(orm):
    orm["Enterprise"].objects = FakeEnterpriseManager([])

    result = _run()

    assert result["success"] is False
    assert "Empresa no encontrada" in result["error"]


def test_calculate_supplier_risk_fails_without_suppliers(orm):
    orm["Enterprise"].objects = FakeEnterpriseManager([FakeEnterprise("e1", "Acme")])
    orm["Suppliers"].reset([])

    result = _run()

    assert result["success"] is False
    assert "No se encontraron suppliers" in result["error"]


def test_calculate_supplier_risk_creates_enterprise_risk(escenario):
    result = _run()

    assert result["success"] is True
    assert result["enterprise_name"] == "Acme"
    assert result["saved"] == 1
    assert result["total_farm_risks"] == 1
    assert result["period_results"][0]["status"] == "created"

    guardado = escenario["EnterpriseRisk"]._store[0]
    assert [fr.id for fr in guardado.risk_input] == ["fr1"]
    assert guardado.risk_output == []


def test_calculate_supplier_risk_updates_existing_enterprise_risk(escenario):
    existente = escenario["EnterpriseRisk"](
        id="er1", enterprise_id="e1", analysis_id="an1", risk_input=[]
    )
    escenario["EnterpriseRisk"].reset([existente])

    result = _run()

    assert result["saved"] == 1
    assert result["period_results"][0]["status"] == "updated"
    assert result["period_results"][0]["new_refs"] == 1
    assert [fr.id for fr in existente.risk_input] == ["fr1"]


def test_calculate_supplier_risk_reports_no_change_when_already_linked(escenario):
    ya_vinculado = escenario["FarmRisk"]._store[0]
    existente = escenario["EnterpriseRisk"](
        id="er1", enterprise_id="e1", analysis_id="an1", risk_input=[ya_vinculado]
    )
    escenario["EnterpriseRisk"].reset([existente])

    result = _run()

    assert result["saved"] == 0
    assert result["period_results"][0]["status"] == "no_change"


def test_calculate_supplier_risk_initialises_null_risk_input(escenario):
    existente = escenario["EnterpriseRisk"](
        id="er1", enterprise_id="e1", analysis_id="an1", risk_input=None
    )
    escenario["EnterpriseRisk"].reset([existente])

    result = _run()

    assert result["period_results"][0]["status"] == "updated"
    assert len(existente.risk_input) == 1


def test_calculate_supplier_risk_dry_run_does_not_write(escenario):
    result = _run(dry_run=True)

    assert result["period_results"][0]["status"] == "dry_run"
    assert result["saved"] == 0
    assert escenario["EnterpriseRisk"]._store == []


def test_calculate_supplier_risk_skips_period_without_suppliers(escenario):
    result = _run(periods=["basura"])

    assert result["skipped"] == 1
    assert result["period_results"][0]["reason"] == "sin suppliers activos para este año"


def test_calculate_supplier_risk_skips_period_without_analysis(escenario):
    escenario["Analysis"].reset([])

    result = _run()

    assert result["skipped"] == 1
    assert result["period_results"][0]["reason"] == "Analysis no encontrado"


def test_calculate_supplier_risk_skips_period_without_farm_risks(escenario):
    escenario["FarmRisk"].reset([])

    result = _run()

    assert result["skipped"] == 1
    assert "sin FarmRisk en BD" in result["period_results"][0]["reason"]


def test_calculate_supplier_risk_skips_farm_risks_without_active_risk(escenario):
    escenario["FarmRisk"].reset(
        [
            escenario["FarmRisk"](
                id="fr1", farm_id=Ref("f1"), analysis_id="an1",
                risk_direct=False, risk_input=False, risk_output=False,
            )
        ]
    )

    result = _run()

    assert result["skipped"] == 1
    assert "ninguno con riesgo activo" in result["period_results"][0]["reason"]


def test_calculate_supplier_risk_writes_package_csv(escenario, tmp_path):
    result = _run(total_risk_dir=str(tmp_path))

    assert result["success"] is True
    generado = tmp_path / "smbyc_supplier_risk_annual_2017_Acme.csv"
    assert generado.exists()


def test_calculate_supplier_risk_tolerates_package_failures(escenario, monkeypatch):
    monkeypatch.setattr(
        supplier_alert,
        "pkg_supplier_risk",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("paquete roto")),
    )

    # El fallo del paquete es informativo: el guardado en BD debe continuar.
    result = _run()

    assert result["saved"] == 1


def test_calculate_supplier_risk_records_save_errors(escenario):
    escenario["EnterpriseRisk"].save_error = RuntimeError("escritura rechazada")

    result = _run()

    assert result["saved"] == 0
    assert result["period_results"][0]["status"] == "error"
    assert "escritura rechazada" in result["period_results"][0]["error"]


def test_calculate_supplier_risk_handles_plain_farm_reference(escenario):
    escenario["FarmRisk"].reset(
        [
            escenario["FarmRisk"](
                id="fr1", farm_id="f1", analysis_id="an1",
                risk_direct=True, risk_input=False, risk_output=False,
            )
        ]
    )

    result = _run()

    assert result["saved"] == 1


def test_calculate_supplier_risk_prints_skipped_and_error_details(escenario, capsys):
    # Mezcla de períodos: uno crea, varios se omiten.
    result = _run(periods=["2017", "aa", "bb", "cc", "dd", "ee", "ff"])

    salida = capsys.readouterr().out
    assert "Omitidos: 6" in salida
    assert "Creados nuevos: 1" in salida
    assert result["periods_processed"] == 7
