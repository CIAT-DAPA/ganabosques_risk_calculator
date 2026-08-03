from types import SimpleNamespace

from main import generate_year_ranges, parse_steps, validate_prerequisites


def test_parse_steps_supports_ranges_and_all():
    assert parse_steps(["1-3", "5"]) == [1, 2, 3, 5]
    assert parse_steps(["all"]) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert parse_steps(["0", "bad", "7-9"]) == [7, 8, 9]


def test_validate_prerequisites_and_generate_year_ranges(tmp_path):
    geojson_folder = tmp_path / "geojsons"
    geojson_folder.mkdir()
    (geojson_folder / "farm.geojson").write_text("{}", encoding="utf-8")

    movements_dir = tmp_path / "movements"
    movements_dir.mkdir()
    (movements_dir / "movement_data_base_2024.csv").write_text("id\n1\n", encoding="utf-8")

    results_dir = tmp_path / "results" / "smbyc" / "annual" / "direct_alerts"
    results_dir.mkdir(parents=True)

    assert validate_prerequisites("direct", None, [], ["2024"], workspace_dir=tmp_path, offline_mode=True, geojsons_folder=str(geojson_folder))[0] is True
    assert validate_prerequisites("movement", None, [], ["2024"], workspace_dir=tmp_path)[0] is True
    assert validate_prerequisites("enterprise", None, [], ["2024"], workspace_dir=tmp_path)[0] is True

    available_periods = [(None, None, "smbyc_deforestation_cumulative_2010-2012", None)]
    assert generate_year_ranges("2010-2012", "smbyc", "cumulative", available_periods) == ["2010-2012"]
    assert generate_year_ranges("2024", "smbyc", "nad", [(None, None, "smbyc_deforestation_nad_202401", None)]) == ["202401"]
