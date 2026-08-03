from types import SimpleNamespace

import pandas as pd
from bson import ObjectId

import supplier_alert


def test_get_years_for_period_and_expand_year_to_quarters():
    assert supplier_alert.get_years_for_period("2017", "annual") == {2017}
    assert supplier_alert.get_years_for_period("2010-2012", "cumulative") == {2010, 2011, 2012}
    assert supplier_alert.expand_year_to_quarters(2024) == ["202401", "202402", "202403", "202404"]


def test_find_enterprise_without_orm_returns_none(monkeypatch):
    monkeypatch.setattr(supplier_alert, "HAS_ORM", False)
    assert supplier_alert.find_enterprise("Acme") is None


def test_supplier_helpers_and_calculation_flow(monkeypatch):
    assert supplier_alert.get_suppliers_for_period([{"farm_id": "farm1", "years": [2017]}], "2017", "annual") == ["farm1"]
    assert supplier_alert.get_analysis_for_period("2017", "annual", "smbyc", "livestock") is None
    assert supplier_alert.find_farm_risks_for_farms([], "analysis") == []

    monkeypatch.setattr(supplier_alert, "HAS_ORM", True)
    monkeypatch.setattr(supplier_alert, "find_enterprise", lambda name_or_code, value_chain=None: SimpleNamespace(id="ent1", name="Acme", type_enterprise="company"))
    monkeypatch.setattr(supplier_alert, "load_suppliers_for_enterprise", lambda enterprise_id, cache_dir=None, force_reload=False: [{"farm_id": "farm1", "years": [2017]}])
    monkeypatch.setattr(supplier_alert, "get_analysis_for_period", lambda period, period_type, source, value_chain: "analysis1")
    monkeypatch.setattr(supplier_alert, "find_farm_risks_for_farms", lambda farm_ids, analysis_id: [SimpleNamespace(id=ObjectId(), farm_id=SimpleNamespace(id="farm1"), risk_direct=True, risk_input=False, risk_output=False)])
    monkeypatch.setattr(supplier_alert, "pkg_supplier_risk", lambda **kwargs: pd.DataFrame([{"id": "farm1"}]))

    class DummyAnalysis:
        class objects:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def first(self):
                return SimpleNamespace(id=ObjectId())

    class DummyEnterpriseRisk:
        class objects:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def first(self):
                return None

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

        def save(self):
            self.saved = True

    monkeypatch.setattr(supplier_alert, "Analysis", DummyAnalysis)
    monkeypatch.setattr(supplier_alert, "EnterpriseRisk", DummyEnterpriseRisk)
    monkeypatch.setattr(supplier_alert, "ObjectId", lambda value=None: value)

    result = supplier_alert.calculate_and_save_supplier_risk("Acme", ["2017"], "annual", dry_run=False)
    assert result["success"] is True
    assert result["saved"] == 1
    assert result["period_results"][0]["status"] == "created"
