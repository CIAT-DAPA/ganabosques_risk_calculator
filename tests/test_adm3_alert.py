from collections import defaultdict

import adm3_alert


def test_build_farm_to_adm3_map_and_total_farms():
    farms_metadata = [
        {"mongo_id": "farm-1", "adm3_id": "adm-1"},
        {"mongo_id": "farm-2", "adm3_id": "adm-1"},
        {"mongo_id": "farm-3", "adm3_id": "adm-2"},
    ]

    farm_map = adm3_alert.build_farm_to_adm3_map(farms_metadata)
    totals = adm3_alert.build_adm3_total_farms(farms_metadata)

    assert farm_map == {"farm-1": "adm-1", "farm-2": "adm-1", "farm-3": "adm-2"}
    assert totals == {"adm-1": 2, "adm-2": 1}


def test_calculate_adm3_risk_for_period_returns_empty_without_orm():
    result = adm3_alert.calculate_adm3_risk_for_period("analysis-1", {}, ["adm-1"], {})
    assert result == []
