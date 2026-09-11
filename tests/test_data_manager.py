import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

import bson

import data_manager
from data_manager import DataManager
from orm_doubles import Ref, make_document


CRS = "EPSG:3116"

# Varias funciones de data_manager importan bson.ObjectId dentro del cuerpo, de modo
# que el doble del módulo no las alcanza: para esas rutas hacen falta ObjectId reales.
OID_ANALYSIS = "507f1f77bcf86cd799439011"
OID_FARM = "507f1f77bcf86cd799439012"
OID_POLYGON = "507f1f77bcf86cd799439013"
OID_DEFO = "507f1f77bcf86cd799439014"
OID_ENTERPRISE = "507f1f77bcf86cd799439015"
OID_USER = "507f1f77bcf86cd799439016"


# ===================== dobles del ORM =====================


class Ext:
    """Sub-documento ext_id: fuente + código externo."""

    def __init__(self, source, ext_code, label=None):
        self.source = source
        self.ext_code = ext_code
        self.label = label


class SourceEnum:
    """Imita un Enum de mongoengine con atributo .value."""

    def __init__(self, value):
        self.value = value


class FarmObjectsManager:
    """Resuelve los filtros sobre ext_id que usa data_manager."""

    def __init__(self, store):
        self.store = store

    def _filtrar(self, filters):
        items = list(self.store)

        vc = filters.get("value_chain")
        if vc is not None:
            items = [f for f in items if getattr(f, "value_chain", None) == vc]

        source = filters.get("ext_id__source")
        code = filters.get("ext_id__ext_code")
        codes = filters.get("ext_id__ext_code__in")

        def coincide(farm):
            for ext in getattr(farm, "ext_id", None) or []:
                src = getattr(ext.source, "value", ext.source)
                if source is not None and src != source:
                    continue
                if code is not None and str(ext.ext_code) != str(code):
                    continue
                if codes is not None and str(ext.ext_code) not in {str(c) for c in codes}:
                    continue
                return True
            return False

        if source is not None or code is not None or codes is not None:
            items = [f for f in items if coincide(f)]
        return items

    def __call__(self, **filters):
        return _QuerySet(self._filtrar(filters))

    def count(self):
        return len(self.store)

    def only(self, *fields):
        return _QuerySet(list(self.store))

    def order_by(self, *fields):
        return _QuerySet(list(self.store))

    def __iter__(self):
        return iter(self.store)


class EnterpriseObjectsManager:
    def __init__(self, store):
        self.store = store

    def __call__(self, **filters):
        label = filters.get("ext_id__label")
        code = filters.get("ext_id__ext_code")
        tipo = filters.get("type_enterprise")

        def coincide(ent):
            if tipo is not None and ent.type_enterprise != tipo:
                return False
            for ext in ent.ext_id or []:
                if label is not None and ext.label != label:
                    continue
                if code is not None and str(ext.ext_code) != str(code):
                    continue
                return True
            return False

        return _QuerySet([e for e in self.store if coincide(e)])


class _QuerySet:
    def __init__(self, items):
        self._items = list(items)

    def only(self, *fields):
        return self

    def order_by(self, *fields):
        return self

    def select_related(self):
        return self

    def first(self):
        return self._items[0] if self._items else None

    def count(self):
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)


@pytest.fixture
def orm(monkeypatch):
    """Instala dobles para todas las colecciones que usa DataManager."""
    Farm = make_document("Farm")
    FarmPolygons = make_document("FarmPolygons")
    Deforestation = make_document("Deforestation")
    Analysis = make_document("Analysis")
    FarmRisk = make_document("FarmRisk")
    EnterpriseRisk = make_document("EnterpriseRisk")
    Enterprise = make_document("Enterprise")
    ProtectedAreas = make_document("ProtectedAreas")
    FarmingAreas = make_document("FarmingAreas")

    Farm.objects = FarmObjectsManager(Farm._store)
    Enterprise.objects = EnterpriseObjectsManager(Enterprise._store)

    monkeypatch.setattr(data_manager, "HAS_ORM", True)
    for nombre, clase in {
        "Farm": Farm,
        "FarmPolygons": FarmPolygons,
        "Deforestation": Deforestation,
        "Analysis": Analysis,
        "FarmRisk": FarmRisk,
        "EnterpriseRisk": EnterpriseRisk,
        "Enterprise": Enterprise,
        "ProtectedAreas": ProtectedAreas,
        "FarmingAreas": FarmingAreas,
    }.items():
        monkeypatch.setattr(data_manager, nombre, clase, raising=False)

    # ObjectId de mentira: conserva el string para poder comparar en asserts.
    monkeypatch.setattr(data_manager, "ObjectId", lambda v=None: str(v), raising=False)

    return {
        "Farm": Farm,
        "FarmPolygons": FarmPolygons,
        "Deforestation": Deforestation,
        "Analysis": Analysis,
        "FarmRisk": FarmRisk,
        "EnterpriseRisk": EnterpriseRisk,
        "Enterprise": Enterprise,
        "ProtectedAreas": ProtectedAreas,
        "FarmingAreas": FarmingAreas,
    }


@pytest.fixture
def no_orm(monkeypatch):
    monkeypatch.setattr(data_manager, "HAS_ORM", False)


@pytest.fixture
def dm(tmp_path):
    return DataManager(str(tmp_path), "http://geo.example", "user", "pass")


# ===================== inicialización y utilidades =====================


def test_data_manager_creates_workspace_layout(tmp_path):
    manager = DataManager(str(tmp_path), "http://geo.example/", "u", "p")

    assert manager.workspace_dir == tmp_path / "alertas"
    assert manager.geojsons_dir.is_dir()
    assert manager.rasters_dir.is_dir()
    assert manager.results_dir.is_dir()
    # La barra final de la URL se recorta para poder concatenar rutas.
    assert manager.gs_url == "http://geo.example"


def test_geojson_stems_cache_is_lazy_and_invalidable(dm):
    assert dm._list_local_geojson_stems() == set()

    (dm.geojsons_dir / "A1.geojson").write_text("{}", encoding="utf-8")
    # Sigue devolviendo el valor memorizado hasta invalidar.
    assert dm._list_local_geojson_stems() == set()

    dm._invalidate_geojson_stems_cache()
    assert dm._list_local_geojson_stems() == {"A1"}


def test_chunked_splits_lists():
    assert DataManager._chunked([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert DataManager._chunked([], 3) == []


def test_farm_ref_to_str_handles_references_and_plain_values():
    assert DataManager._farm_ref_to_str(SimpleNamespace(id="abc")) == "abc"
    assert DataManager._farm_ref_to_str("xyz") == "xyz"


def test_normalize_external_farm_id():
    assert DataManager._normalize_external_farm_id("FARM_ID_00123") == "00123"
    assert DataManager._normalize_external_farm_id("farm_id_x") == "x"
    assert DataManager._normalize_external_farm_id("  123  ") == "123"
    assert DataManager._normalize_external_farm_id(None) == ""
    assert DataManager._normalize_external_farm_id("nan") == ""
    assert DataManager._normalize_external_farm_id("None") == ""
    assert DataManager._normalize_external_farm_id("") == ""


def test_get_farmrisk_cache_for_analysis(dm):
    assert dm.get_farmrisk_cache_for_analysis(None) == {}
    assert dm.get_farmrisk_cache_for_analysis("desconocido") == {}

    dm._farmrisk_cache_by_analysis["42"] = {"123": "fr1"}
    assert dm.get_farmrisk_cache_for_analysis("42") == {"123": "fr1"}
    assert dm.get_farmrisk_cache_for_analysis(42) == {"123": "fr1"}


# ===================== mapas de polígonos =====================


def test_build_preferred_polygon_map_prefers_enabled(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset(
        [
            FarmPolygons(id="poly_off", farm_id=Ref("f1"), log=SimpleNamespace(enable=False)),
            FarmPolygons(id="poly_on", farm_id=Ref("f1"), log=SimpleNamespace(enable=True)),
        ]
    )

    class Manager:
        def __call__(self, **filters):
            items = [
                p for p in FarmPolygons._store
                if str(p.farm_id.id) in {str(getattr(f, "id", f)) for f in filters["farm_id__in"]}
            ]
            if filters.get("log__enable"):
                items = [p for p in items if p.log.enable]
            return _QuerySet(items)

    FarmPolygons.objects = Manager()

    assert dm._build_preferred_polygon_map(["f1"]) == {"f1": "poly_on"}


def test_build_preferred_polygon_map_falls_back_to_any(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset(
        [FarmPolygons(id="poly_off", farm_id=Ref("f1"), log=SimpleNamespace(enable=False))]
    )

    class Manager:
        def __call__(self, **filters):
            items = list(FarmPolygons._store)
            if filters.get("log__enable"):
                items = [p for p in items if p.log.enable]
            return _QuerySet(items)

    FarmPolygons.objects = Manager()

    assert dm._build_preferred_polygon_map(["f1"]) == {"f1": "poly_off"}


def test_build_preferred_polygon_map_returns_empty_without_ids_or_orm(dm, orm, monkeypatch):
    assert dm._build_preferred_polygon_map([]) == {}

    monkeypatch.setattr(data_manager, "HAS_ORM", False)
    assert dm._build_preferred_polygon_map(["f1"]) == {}


def test_build_preferred_polygon_map_survives_query_errors(dm, orm):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    orm["FarmPolygons"].objects = Manager()

    assert dm._build_preferred_polygon_map(["f1"]) == {}


def test_get_preferred_farm_polygon(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    habilitado = FarmPolygons(id="p_on", farm_id="f1", log=SimpleNamespace(enable=True))
    deshabilitado = FarmPolygons(id="p_off", farm_id="f1", log=SimpleNamespace(enable=False))
    FarmPolygons.reset([deshabilitado, habilitado])

    class Manager:
        def __call__(self, **filters):
            items = [p for p in FarmPolygons._store if p.farm_id == filters["farm_id"]]
            if filters.get("log__enable"):
                items = [p for p in items if p.log.enable]
            return _QuerySet(items)

    FarmPolygons.objects = Manager()

    assert dm._get_preferred_farm_polygon("f1") is habilitado


def test_get_preferred_farm_polygon_without_orm(dm, no_orm):
    assert dm._get_preferred_farm_polygon("f1") is None


def test_get_preferred_farm_polygon_returns_none_on_error(dm, orm):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    orm["FarmPolygons"].objects = Manager()

    assert dm._get_preferred_farm_polygon("f1") is None


# ===================== load_farms_metadata =====================


def _farm(farm_id, sit=None, geo=None, prod=None, adm3=None, value_chain=None):
    exts = []
    if sit:
        exts.append(Ext(SourceEnum("SIT_CODE"), sit))
    if geo:
        exts.append(Ext(SourceEnum("GEOFARMER_ID"), geo))
    if prod:
        exts.append(Ext(SourceEnum("PRODUCER_ID"), prod))
    return SimpleNamespace(
        id=farm_id,
        ext_id=exts,
        adm3_id=Ref(adm3) if adm3 else None,
        value_chain=value_chain,
    )


def test_load_farms_metadata_reads_from_mongo_and_caches(dm, orm, monkeypatch):
    orm["Farm"]._store.extend(
        [
            _farm("f1", sit="S1", prod="P1", adm3="a1"),
            _farm("f2", geo="G2"),
        ]
    )
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {"f1": "poly1"})

    metadata, error = dm.load_farms_metadata()

    assert error is None
    assert metadata[0] == {
        "mongo_id": "f1",
        "sitcode": "S1",
        "sit_code": "S1",
        "geofarmer_id": None,
        "producer_id": "P1",
        "adm3_id": "a1",
        "farm_polygon_id": "poly1",
    }
    # Sin SIT_CODE, livestock cae en GEOFARMER_ID.
    assert metadata[1]["sitcode"] == "G2"
    assert metadata[1]["farm_polygon_id"] is None
    # El caché JSON queda escrito para la próxima corrida.
    assert (dm.farms_dir / "farms_metadata.json").exists()


def test_load_farms_metadata_prefers_geofarmer_for_cacao(dm, orm, monkeypatch):
    from ganabosques_orm.enums.valuechain import ValueChain

    orm["Farm"]._store.append(
        _farm("f1", sit="S1", geo="G1", value_chain=ValueChain.CACAO)
    )
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {})

    metadata, error = dm.load_farms_metadata(value_chain="cacao")

    assert error is None
    assert metadata[0]["sitcode"] == "G1"


def test_load_farms_metadata_rejects_unknown_value_chain(dm, orm):
    metadata, error = dm.load_farms_metadata(value_chain="inexistente")

    assert metadata == []
    assert "no válido" in error


def test_load_farms_metadata_returns_memory_cache(dm):
    dm._farms_metadata = [{"mongo_id": "f1"}, {"mongo_id": "f2"}]

    assert dm.load_farms_metadata() == (dm._farms_metadata, None)
    # Con límite se recorta sin volver a consultar.
    recortado, error = dm.load_farms_metadata(limit=1)
    assert (len(recortado), error) == (1, None)


def test_load_farms_metadata_reads_json_cache(dm):
    datos = [{"mongo_id": "f1", "sitcode": "S1"}, {"mongo_id": "f2", "sitcode": "S2"}]
    (dm.farms_dir).mkdir(parents=True, exist_ok=True)
    (dm.farms_dir / "farms_metadata.json").write_text(json.dumps(datos), encoding="utf-8")

    metadata, error = dm.load_farms_metadata()

    assert (metadata, error) == (datos, None)


def test_load_farms_metadata_applies_limit_to_json_cache(dm):
    datos = [{"mongo_id": f"f{i}"} for i in range(5)]
    (dm.farms_dir).mkdir(parents=True, exist_ok=True)
    (dm.farms_dir / "farms_metadata.json").write_text(json.dumps(datos), encoding="utf-8")

    metadata, _ = dm.load_farms_metadata(limit=2)

    assert len(metadata) == 2


def test_load_farms_metadata_uses_value_chain_specific_cache(dm):
    (dm.farms_dir).mkdir(parents=True, exist_ok=True)
    (dm.farms_dir / "farms_metadata_cacao.json").write_text(
        json.dumps([{"mongo_id": "f_cacao"}]), encoding="utf-8"
    )

    metadata, _ = dm.load_farms_metadata(value_chain="cacao")

    assert metadata == [{"mongo_id": "f_cacao"}]


def test_load_farms_metadata_falls_back_to_mongo_on_corrupt_cache(dm, orm, monkeypatch, capsys):
    (dm.farms_dir).mkdir(parents=True, exist_ok=True)
    (dm.farms_dir / "farms_metadata.json").write_text("{ roto", encoding="utf-8")
    orm["Farm"]._store.append(_farm("f1", sit="S1"))
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {})

    metadata, error = dm.load_farms_metadata()

    assert error is None
    assert metadata[0]["mongo_id"] == "f1"
    assert "Error leyendo caché" in capsys.readouterr().out


def test_load_farms_metadata_ignores_cache_when_refreshing(dm, orm, monkeypatch, capsys):
    (dm.farms_dir).mkdir(parents=True, exist_ok=True)
    (dm.farms_dir / "farms_metadata.json").write_text(
        json.dumps([{"mongo_id": "viejo"}]), encoding="utf-8"
    )
    orm["Farm"]._store.append(_farm("f1", sit="S1"))
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {})

    metadata, _ = dm.load_farms_metadata(refresh_data=True)

    assert metadata[0]["mongo_id"] == "f1"
    assert "ignorando caché" in capsys.readouterr().out


def test_load_farms_metadata_offline_without_cache_returns_error(dm, orm):
    metadata, error = dm.load_farms_metadata(offline_mode=True)

    assert metadata == []
    assert "Modo offline" in error


def test_load_farms_metadata_without_orm(dm, no_orm):
    metadata, error = dm.load_farms_metadata()

    assert metadata == []
    assert error == "ganabosques_orm no está disponible"


def test_load_farms_metadata_reports_connection_errors(dm, orm):
    class Explota:
        def count(self):
            raise RuntimeError("sin conexión")

        def only(self, *a):
            raise RuntimeError("sin conexión")

    orm["Farm"].objects = Explota()

    metadata, error = dm.load_farms_metadata()

    assert metadata == []
    assert "No se pudo conectar a MongoDB" in error


def test_load_farms_metadata_stops_at_limit(dm, orm, monkeypatch, capsys):
    orm["Farm"]._store.extend([_farm(f"f{i}", sit=f"S{i}") for i in range(5)])
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {})

    metadata, _ = dm.load_farms_metadata(limit=2)

    assert len(metadata) == 2
    assert "Límite alcanzado" in capsys.readouterr().out
    # Con límite no se persiste caché parcial.
    assert not (dm.farms_dir / "farms_metadata.json").exists()


def test_load_farms_metadata_reports_empty_database(dm, orm, monkeypatch):
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {})

    metadata, error = dm.load_farms_metadata()

    assert metadata == []
    assert error == "No se encontraron farms en la base de datos"


def test_load_farms_metadata_skips_broken_ext_ids(dm, orm, monkeypatch):
    class ExtRoto:
        @property
        def source(self):
            raise RuntimeError("ext corrupto")

        ext_code = "x"

    farm = SimpleNamespace(id="f1", ext_id=[ExtRoto(), Ext(SourceEnum("SIT_CODE"), "S1")], adm3_id=None)
    orm["Farm"]._store.append(farm)
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {})

    metadata, error = dm.load_farms_metadata()

    assert error is None
    assert metadata[0]["sit_code"] == "S1"


def test_load_farms_metadata_survives_polygon_lookup_errors(dm, orm, monkeypatch, capsys):
    orm["Farm"]._store.append(_farm("f1", sit="S1"))
    monkeypatch.setattr(
        dm, "_build_preferred_polygon_map",
        lambda ids: (_ for _ in ()).throw(RuntimeError("polygons caídos")),
    )

    metadata, error = dm.load_farms_metadata()

    assert error is None
    assert metadata[0]["farm_polygon_id"] is None
    assert "No se pudieron cargar farm_polygon_ids" in capsys.readouterr().out


def test_load_farms_metadata_reports_cache_write_errors(dm, orm, monkeypatch, capsys):
    orm["Farm"]._store.append(_farm("f1", sit="S1"))
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {})
    monkeypatch.setattr(
        data_manager.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disco lleno"))
    )

    metadata, error = dm.load_farms_metadata()

    assert error is None
    assert "No se pudo guardar caché" in capsys.readouterr().out


# ===================== geojsons =====================


def test_count_available_geojsons(dm, capsys):
    (dm.geojsons_dir / "A1.geojson").write_text("{}", encoding="utf-8")

    total = dm.count_available_geojsons(
        [{"sitcode": "A1"}, {"sitcode": "NO_EXISTE"}, {"sitcode": None}]
    )

    assert total == 1
    assert "1 encontrados, 2 faltantes" in capsys.readouterr().out


def _polygon_doc(FarmPolygons, farm_id, geom_id="A1"):
    contenido = json.dumps(
        {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"n": geom_id}, "geometry": None}]}
    )
    return FarmPolygons(id=f"poly_{farm_id}", farm_id=Ref(farm_id), geojson=contenido)


def _polygons_manager(FarmPolygons):
    class Manager:
        def __call__(self, **filters):
            items = list(FarmPolygons._store)
            if "farm_id__in" in filters:
                claves = {str(x) for x in filters["farm_id__in"]}
                items = [p for p in items if str(p.farm_id.id) in claves]
            if "farm_id" in filters:
                items = [p for p in items if str(p.farm_id.id) == str(filters["farm_id"])]
            return _QuerySet(items)

    return Manager()


def test_prepare_geojsons_counts_local_files(dm, orm):
    (dm.geojsons_dir / "A1.geojson").write_text("{}", encoding="utf-8")
    orm["FarmPolygons"].objects = _polygons_manager(orm["FarmPolygons"])

    total = dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": "A1"}])

    assert total == 1


def test_prepare_geojsons_downloads_missing(dm, orm, capsys):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([_polygon_doc(FarmPolygons, "f1")])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)

    total = dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": "A1"}])

    assert total == 1
    assert (dm.geojsons_dir / "A1.geojson").exists()
    assert "1 descargados" in capsys.readouterr().out


def test_prepare_geojsons_counts_farms_without_sitcode(dm, orm):
    orm["FarmPolygons"].objects = _polygons_manager(orm["FarmPolygons"])

    total = dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": None}])

    assert total == 0


def test_prepare_geojsons_force_download_refreshes_everything(dm, orm, capsys):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([_polygon_doc(FarmPolygons, "f1", geom_id="NUEVO")])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)
    (dm.geojsons_dir / "A1.geojson").write_text('{"viejo": true}', encoding="utf-8")

    total = dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": "A1"}], force_download=True)

    assert total == 1
    contenido = json.loads((dm.geojsons_dir / "A1.geojson").read_text(encoding="utf-8"))
    assert contenido["features"][0]["properties"]["n"] == "NUEVO"
    assert "force_download=True" in capsys.readouterr().out


def test_prepare_geojsons_force_download_counts_incomplete_records(dm, orm):
    orm["FarmPolygons"].objects = _polygons_manager(orm["FarmPolygons"])

    total = dm.prepare_geojsons(
        [{"mongo_id": None, "sitcode": "A1"}, {"mongo_id": "f2", "sitcode": None}],
        force_download=True,
    )

    assert total == 0


def test_prepare_geojsons_skips_documents_without_geojson(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([FarmPolygons(id="p1", farm_id=Ref("f1"), geojson=None)])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)

    assert dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": "A1"}]) == 0


def test_prepare_geojsons_handles_invalid_json(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([FarmPolygons(id="p1", farm_id=Ref("f1"), geojson="{ roto")])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)

    assert dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": "A1"}]) == 0


def test_prepare_geojsons_ignores_invalid_object_ids(dm, orm, monkeypatch):
    monkeypatch.setattr(
        data_manager, "ObjectId", lambda v=None: (_ for _ in ()).throw(ValueError("oid malo"))
    )
    orm["FarmPolygons"].objects = _polygons_manager(orm["FarmPolygons"])

    assert dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": "A1"}]) == 0


def test_prepare_geojsons_survives_batch_query_errors(dm, orm):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    orm["FarmPolygons"].objects = Manager()

    assert dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": "A1"}]) == 0


def test_prepare_geojsons_without_orm(dm, no_orm):
    assert dm.prepare_geojsons([{"mongo_id": "f1", "sitcode": "A1"}]) == 0


def test_ensure_geojson_available_returns_cached_path(dm):
    ruta = dm.geojsons_dir / "A1.geojson"
    ruta.write_text("{}", encoding="utf-8")

    assert dm.ensure_geojson_available({"mongo_id": OID_FARM, "sitcode": "A1"}) == str(ruta)


def test_ensure_geojson_available_requires_sitcode(dm):
    assert dm.ensure_geojson_available({"mongo_id": "f1", "sitcode": None}) is None


def test_ensure_geojson_available_downloads_when_missing(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([_polygon_doc(FarmPolygons, OID_FARM)])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)

    ruta = dm.ensure_geojson_available({"mongo_id": OID_FARM, "sitcode": "A1"})

    assert ruta == str(dm.geojsons_dir / "A1.geojson")


def test_download_geojson_without_orm(dm, no_orm):
    assert dm._download_geojson(OID_FARM, "A1") is None


def test_download_geojson_uses_mongo_id_without_sitcode(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([_polygon_doc(FarmPolygons, OID_FARM)])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)

    ruta = dm._download_geojson(OID_FARM, None)

    assert Path(ruta).name == f"{OID_FARM}.geojson"


def test_download_geojson_returns_none_when_not_found(dm, orm):
    orm["FarmPolygons"].objects = _polygons_manager(orm["FarmPolygons"])

    assert dm._download_geojson(OID_FARM, "A1") is None


def test_download_geojson_returns_none_on_error(dm, orm, monkeypatch):
    monkeypatch.setattr(
        data_manager, "ObjectId", lambda v=None: (_ for _ in ()).throw(ValueError("oid malo"))
    )

    assert dm._download_geojson(OID_FARM, "A1") is None


# ===================== sanitización de geojson =====================


def test_strip_unsupported_properties_removes_centroid(dm):
    datos = {
        "type": "FeatureCollection",
        "features": [{"type": "Feature", "properties": {"centroid": [1, 2], "n": "x"}}],
    }

    assert dm._strip_unsupported_properties(datos) is True
    assert "centroid" not in datos["features"][0]["properties"]


def test_strip_unsupported_properties_handles_single_feature(dm):
    datos = {"type": "Feature", "properties": {"centroid": [1, 2]}}

    assert dm._strip_unsupported_properties(datos) is True


def test_strip_unsupported_properties_returns_false_for_other_inputs(dm):
    assert dm._strip_unsupported_properties("no es un dict") is False
    assert dm._strip_unsupported_properties({"type": "Point"}) is False
    assert dm._strip_unsupported_properties(
        {"type": "FeatureCollection", "features": ["no es un dict"]}
    ) is False
    assert dm._strip_unsupported_properties(
        {"type": "Feature", "properties": {"n": "x"}}
    ) is False


def test_sanitize_geojson_file_rewrites_only_when_needed(dm):
    ruta = dm.geojsons_dir / "A1.geojson"
    ruta.write_text(
        json.dumps(
            {"type": "Feature", "properties": {"centroid": [1, 2], "n": "x"}}
        ),
        encoding="utf-8",
    )

    dm._sanitize_geojson_file(ruta)

    guardado = json.loads(ruta.read_text(encoding="utf-8"))
    assert "centroid" not in guardado["properties"]
    assert guardado["properties"]["n"] == "x"


def test_sanitize_geojson_file_ignores_unreadable_files(dm):
    ruta = dm.geojsons_dir / "roto.geojson"
    ruta.write_text("{ roto", encoding="utf-8")

    # No debe propagar la excepción.
    dm._sanitize_geojson_file(ruta)


# ===================== rasters =====================


class FakeResponse:
    def __init__(self, status_code=200, content_type="image/geotiff", payload=b"tiff", text=""):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.text = text
        self._payload = payload

    def iter_content(self, chunk_size=8192):
        yield self._payload


def test_ensure_raster_available_returns_cached(dm, monkeypatch, capsys):
    destino = dm.rasters_dir / "smbyc" / "annual"
    destino.mkdir(parents=True)
    (destino / "capa_2024.tif").write_bytes(b"x")

    monkeypatch.setattr(
        data_manager.requests, "get", lambda *a, **k: pytest.fail("no debería descargar")
    )

    ruta = dm.ensure_raster_available("smbyc", "annual", "capa", "capa_2024", "2024")

    assert ruta == str(destino / "capa_2024.tif")
    assert "en caché" in capsys.readouterr().out


def test_ensure_raster_available_downloads(dm, monkeypatch, capsys):
    monkeypatch.setattr(data_manager.requests, "get", lambda *a, **k: FakeResponse())

    ruta = dm.ensure_raster_available("smbyc", "annual", "capa", "capa_2024", "2024")

    assert ruta is not None
    assert Path(ruta).read_bytes() == b"tiff"
    assert "Descargado" in capsys.readouterr().out


def test_download_raster_returns_none_on_http_error(dm, monkeypatch):
    monkeypatch.setattr(
        data_manager.requests, "get", lambda *a, **k: FakeResponse(status_code=500, text="boom")
    )

    assert dm.ensure_raster_available("smbyc", "annual", "capa", "c", "2024") is None


def test_download_raster_returns_none_for_xml_response(dm, monkeypatch):
    monkeypatch.setattr(
        data_manager.requests,
        "get",
        lambda *a, **k: FakeResponse(content_type="application/xml", text="<error/>"),
    )

    assert dm.ensure_raster_available("smbyc", "annual", "capa", "c", "2024") is None


def test_download_raster_cleans_partial_file_on_error(dm, monkeypatch):
    class ResponseQueFalla(FakeResponse):
        def iter_content(self, chunk_size=8192):
            yield b"parcial"
            raise RuntimeError("conexión cortada")

    monkeypatch.setattr(data_manager.requests, "get", lambda *a, **k: ResponseQueFalla())

    assert dm.ensure_raster_available("smbyc", "annual", "capa", "c", "2024") is None
    assert not (dm.rasters_dir / "smbyc" / "annual" / "c.tif").exists()


def test_extract_time_filter_from_name():
    dm_cls = DataManager.extract_time_filter_from_name
    manager = SimpleNamespace()

    assert dm_cls(manager, "smbyc_deforestation_annual_2013-2014") == "2013-2014"
    assert dm_cls(manager, "smbyc_deforestation_cumulative_2010-2017") == "2010-2017"
    assert dm_cls(manager, "nad_deforestation_quarter_202301") == "2023-01"
    # Sin patrón reconocible: último segmento.
    assert dm_cls(manager, "capa_rara_final") == "final"


# ===================== get_mongo_id_map_df / get_results_dir =====================


def test_get_mongo_id_map_df_returns_empty_without_metadata(dm):
    df = dm.get_mongo_id_map_df()

    assert df.empty
    assert list(df.columns) == ["id", "farm_id", "farm_poligons_id", "GEOFARMER_ID"]


def test_get_mongo_id_map_df_builds_rows(dm):
    dm._farms_metadata = [
        {"mongo_id": "f1", "sitcode": "FARM_ID_00123", "geofarmer_id": "G1", "farm_polygon_id": "p1"},
        {"mongo_id": "f2", "sitcode": "S2", "geofarmer_id": None, "farm_polygon_id": None},
        {"mongo_id": "f3", "sitcode": None, "geofarmer_id": None},  # se descarta
    ]

    df = dm.get_mongo_id_map_df()

    assert len(df) == 2
    assert df.iloc[0]["id"] == "00123"
    assert df.iloc[0]["GEOFARMER_ID"] == "G1"
    assert df.iloc[1]["farm_poligons_id"] == ""


def test_get_mongo_id_map_df_applies_filter(dm):
    dm._farms_metadata = [
        {"mongo_id": "f1", "sitcode": "S1", "geofarmer_id": ""},
        {"mongo_id": "f2", "sitcode": "S2", "geofarmer_id": ""},
        {"mongo_id": "f3", "sitcode": "", "geofarmer_id": "G3"},
    ]

    df = dm.get_mongo_id_map_df(ids_filter={"S1", "G3"})

    assert sorted(df["farm_id"]) == ["f1", "f3"]


def test_get_results_dir_new_and_legacy_layouts(dm):
    nuevo = dm.get_results_dir("direct_alerts", source="smbyc", deforestation_type="annual")
    assert nuevo == dm.results_dir / "smbyc" / "annual" / "direct_alerts"
    assert nuevo.is_dir()

    legado = dm.get_results_dir("direct_alerts")
    assert legado == dm.results_dir / "direct_alerts"
    assert legado.is_dir()


# ===================== caché de geometrías =====================


def _write_geojson(path, geom, crs=CRS):
    gpd.GeoDataFrame({"geometry": [geom]}, crs=crs).to_file(path, driver="GeoJSON")


def test_load_geometries_to_cache(dm):
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))

    stats = dm.load_geometries_to_cache(
        [{"sitcode": "A1", "mongo_id": "f1"}, {"sitcode": "NO_EXISTE"}, {"sitcode": None}]
    )

    assert stats["loaded"] == 1
    assert stats["failed"] == 2
    assert stats["cache_size_mb"] > 0
    assert dm.is_geometry_cache_loaded() is True


def test_load_geometries_to_cache_merges_multiple_polygons(dm):
    gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10)]}, crs=CRS
    ).to_file(dm.geojsons_dir / "A1.geojson", driver="GeoJSON")

    dm.load_geometries_to_cache([{"sitcode": "A1"}], show_progress=False)

    assert dm.get_geometry("A1").geom_type == "MultiPolygon"


def test_load_geometries_to_cache_reprojects(dm):
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(-74, 4, -73.99, 4.01), crs="EPSG:4326")

    stats = dm.load_geometries_to_cache([{"sitcode": "A1"}], show_progress=False)

    assert stats["loaded"] == 1


def test_load_geometries_to_cache_skips_empty_layers(dm, monkeypatch):
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))
    monkeypatch.setattr(
        data_manager.gpd,
        "read_file",
        lambda *a, **k: gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS),
    )

    stats = dm.load_geometries_to_cache([{"sitcode": "A1"}], show_progress=False)

    assert stats == {"loaded": 0, "failed": 1, "cache_size_mb": stats["cache_size_mb"]}


def test_load_geometries_to_cache_skips_unreadable_files(dm):
    (dm.geojsons_dir / "A1.geojson").write_text("{ roto", encoding="utf-8")

    stats = dm.load_geometries_to_cache([{"sitcode": "A1"}], show_progress=False)

    assert stats["failed"] == 1


def test_get_all_cached_geometries_and_geodataframe(dm):
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))
    dm.load_geometries_to_cache([{"sitcode": "A1"}], show_progress=False)

    assert set(dm.get_all_cached_geometries()) == {"A1"}
    gdf = dm.get_farms_geodataframe()
    assert gdf.crs.to_string() == CRS
    assert gdf["id"].tolist() == ["A1"]


def test_get_farms_geodataframe_requires_cache(dm):
    with pytest.raises(RuntimeError, match="Caché de geometrías vacío"):
        dm.get_farms_geodataframe()


def test_get_geometry_loads_from_disk_and_caches(dm):
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))

    geom = dm.get_geometry("A1")

    assert geom is not None
    assert "A1" in dm._geometries_cache
    # Segunda llamada: sale del caché.
    assert dm.get_geometry("A1") is geom


def test_get_geometry_merges_multiple_polygons(dm):
    gpd.GeoDataFrame(
        {"geometry": [box(0, 0, 10, 10), box(20, 0, 30, 10)]}, crs=CRS
    ).to_file(dm.geojsons_dir / "A1.geojson", driver="GeoJSON")

    assert dm.get_geometry("A1").geom_type == "MultiPolygon"


def test_get_geometry_reprojects(dm):
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(-74, 4, -73.99, 4.01), crs="EPSG:4326")

    assert dm.get_geometry("A1") is not None


def test_get_geometry_returns_none_for_missing_or_broken_files(dm, monkeypatch):
    assert dm.get_geometry("NO_EXISTE") is None

    (dm.geojsons_dir / "roto.geojson").write_text("{ roto", encoding="utf-8")
    assert dm.get_geometry("roto") is None

    _write_geojson(dm.geojsons_dir / "vacio.geojson", box(0, 0, 1, 1))
    monkeypatch.setattr(
        data_manager.gpd,
        "read_file",
        lambda *a, **k: gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=CRS),
    )
    assert dm.get_geometry("vacio") is None


# ===================== índice espacial =====================


def test_build_spatial_index_and_query(dm):
    _write_geojson(dm.geojsons_dir / "A1.geojson", box(0, 0, 10, 10))
    _write_geojson(dm.geojsons_dir / "A2.geojson", box(100, 100, 110, 110))
    dm.load_geometries_to_cache([{"sitcode": "A1"}, {"sitcode": "A2"}], show_progress=False)

    stats = dm.build_spatial_index(show_progress=False)

    assert stats["indexed"] == 2
    assert dm.query_farms_by_bbox((0, 0, 20, 20)) == ["A1"]


def test_build_spatial_index_requires_geometries(dm):
    with pytest.raises(RuntimeError, match="Primero debes cargar geometrías"):
        dm.build_spatial_index()


def test_build_spatial_index_skips_broken_geometries(dm):
    dm._geometries_cache = {"A1": box(0, 0, 10, 10), "ROTA": object()}

    stats = dm.build_spatial_index(show_progress=True)

    assert stats["indexed"] == 1


def test_query_farms_by_bbox_requires_index(dm):
    with pytest.raises(RuntimeError, match="Primero debes construir índice"):
        dm.query_farms_by_bbox((0, 0, 1, 1))


# ===================== download_movements_to_csv =====================


@pytest.fixture
def movement_module(monkeypatch):
    import ganabosques_orm.collections.movement as mod

    Movement = make_document("Movement")

    class Manager:
        def __call__(self, **filters):
            return _QuerySet(list(Movement._store))

    Movement.objects = Manager()
    monkeypatch.setattr(mod, "Movement", Movement)
    return Movement


def _mov(origen=None, destino=None, ent_origen=None, ent_destino=None, fecha=None):
    return SimpleNamespace(
        farm_id_origin=origen,
        farm_id_destination=destino,
        enterprise_id_origin=ent_origen,
        enterprise_id_destination=ent_destino,
        type_origin=SourceEnum("FARM"),
        type_destination=SourceEnum("SLAUGHTERHOUSE"),
        date=fecha or datetime(2024, 3, 15),
    )


def test_download_movements_writes_csv(dm, orm, movement_module):
    origen = SimpleNamespace(ext_id=[Ext(SourceEnum("SIT_CODE"), "S1"), Ext(SourceEnum("PRODUCER_ID"), "P1")])
    destino = SimpleNamespace(ext_id=[Ext(SourceEnum("SIT_CODE"), "S2")])
    movement_module.reset([_mov(origen=origen, destino=destino)])

    ruta, error = dm.download_movements_to_csv(2024)

    assert error is None
    df = pd.read_csv(ruta, dtype=str)
    assert df.iloc[0]["SIT_CODE_ORIGEN"] == "S1"
    assert df.iloc[0]["SIT_CODE_DESTINO"] == "S2"
    assert df.iloc[0]["PRODUCER_ID_ORIGEN"] == "P1"
    assert df.iloc[0]["DATE"] == "2024-03-15"
    assert df.iloc[0]["TIPO_ORIGEN"] == "FARM"


def test_download_movements_uses_enterprise_ids_as_producers(dm, orm, movement_module):
    movement_module.reset(
        [
            _mov(
                ent_origen=SimpleNamespace(id="ent1"),
                ent_destino=SimpleNamespace(id="ent2"),
            )
        ]
    )

    ruta, _ = dm.download_movements_to_csv(2024)

    df = pd.read_csv(ruta, dtype=str)
    assert df.iloc[0]["PRODUCER_ID_ORIGEN"] == "ent1"
    assert df.iloc[0]["PRODUCER_ID_DESTINO"] == "ent2"


def test_download_movements_returns_cached_file(dm, orm, movement_module, capsys):
    destino = dm.workspace_dir / "movements"
    destino.mkdir(parents=True)
    existente = destino / "movement_data_base_2024.csv"
    existente.write_text("SIT_CODE_ORIGEN\n", encoding="utf-8")

    ruta, error = dm.download_movements_to_csv(2024)

    assert (ruta, error) == (str(existente), None)
    assert "ya en caché" in capsys.readouterr().out


def test_download_movements_reports_empty_year(dm, orm, movement_module):
    movement_module.reset([])

    ruta, error = dm.download_movements_to_csv(2024)

    assert ruta is None
    assert "No hay movimientos" in error


def test_download_movements_accepts_custom_output_dir(dm, orm, movement_module, tmp_path):
    movement_module.reset([_mov()])
    destino = tmp_path / "otra_carpeta"

    ruta, error = dm.download_movements_to_csv(2024, output_dir=destino)

    assert error is None
    assert Path(ruta).parent == destino


def test_download_movements_without_orm(dm, no_orm):
    ruta, error = dm.download_movements_to_csv(2024)

    assert ruta is None
    assert error == "ganabosques_orm no disponible"


def test_download_movements_reports_errors(dm, orm, movement_module, monkeypatch):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    movement_module.objects = Manager()

    ruta, error = dm.download_movements_to_csv(2024)

    assert ruta is None
    assert "Error descargando movimientos" in error


# ===================== save_farm_risk_to_db =====================


def _total_risk_df(**overrides):
    fila = {
        "id": "S1",
        "farm_id": OID_FARM,
        "farm_poligons_id": OID_POLYGON,
        "direct_alert": "True",
        "deforested_ha": "2.5",
        "deforested_prop": "0.25",
        "protected_ha": "0",
        "protected_prop": "0",
        "farming_in_ha": "1",
        "farming_in_prop": "0.1",
        "farming_out_ha": "9",
        "farming_out_prop": "0.9",
        "indirect_alert_in": "True",
        "indirect_alert_out": "no_info",
    }
    fila.update(overrides)
    return pd.DataFrame([fila])


def _analysis_manager(Analysis):
    class Manager:
        def __call__(self, **filters):
            items = [a for a in Analysis._store if str(a.id) == str(filters.get("id"))]
            return _QuerySet(items)

    return Manager()


def _farmrisk_manager(FarmRisk):
    class Manager:
        def __call__(self, **filters):
            items = [
                fr for fr in FarmRisk._store
                if str(getattr(fr, "farm_id", None)) == str(filters.get("farm_id"))
                and str(getattr(fr, "analysis_id", None)) == str(filters.get("analysis_id"))
            ]
            return _QuerySet(items)

        def insert(self, docs, load_bulk=False):
            if FarmRisk.insert_error:
                raise FarmRisk.insert_error
            FarmRisk._store.extend(docs)
            return docs

    return Manager()


@pytest.fixture
def risk_orm(dm, orm):
    Analysis, FarmRisk = orm["Analysis"], orm["FarmRisk"]
    Analysis.reset([Analysis(id=OID_ANALYSIS)])
    Analysis.objects = _analysis_manager(Analysis)
    FarmRisk.reset([])
    FarmRisk.objects = _farmrisk_manager(FarmRisk)
    return orm


def test_save_farm_risk_creates_documents(dm, risk_orm):
    saved, failed, errors = dm.save_farm_risk_to_db(_total_risk_df(), OID_ANALYSIS)

    assert (saved, failed, errors) == (1, 0, [])
    doc = risk_orm["FarmRisk"]._store[0]
    assert doc.deforestation.ha == 2.5
    assert doc.risk_direct is True
    assert doc.risk_input is True
    # 'no_info' se traduce a False, no a un error.
    assert doc.risk_output is False
    assert dm.get_farmrisk_cache_for_analysis(OID_ANALYSIS) == {"S1": doc.id}


def test_save_farm_risk_updates_existing(dm, risk_orm):
    FarmRisk = risk_orm["FarmRisk"]
    existente = FarmRisk(id="fr1", farm_id=OID_FARM, analysis_id=OID_ANALYSIS)
    FarmRisk.reset([existente])
    FarmRisk.objects = _farmrisk_manager(FarmRisk)

    saved, failed, _ = dm.save_farm_risk_to_db(_total_risk_df(), OID_ANALYSIS)

    assert (saved, failed) == (1, 0)
    assert len(FarmRisk._store) == 1
    assert existente.deforestation.ha == 2.5


def test_save_farm_risk_resolves_by_sitcode_when_ids_are_missing(dm, risk_orm, monkeypatch):
    risk_orm["Farm"]._store.append(_farm(OID_FARM, sit="S1"))
    monkeypatch.setattr(
        dm, "_get_preferred_farm_polygon", lambda oid: SimpleNamespace(id="p9")
    )

    saved, failed, _ = dm.save_farm_risk_to_db(
        _total_risk_df(farm_id="", farm_poligons_id=""), OID_ANALYSIS
    )

    assert (saved, failed) == (1, 0)
    assert risk_orm["FarmRisk"]._store[0].farm_polygons_id == "p9"


def test_save_farm_risk_falls_back_to_any_ext_code(dm, risk_orm, monkeypatch):
    # No hay coincidencia por SIT_CODE, pero sí por cualquier ext_code.
    risk_orm["Farm"]._store.append(_farm(OID_FARM, geo="S1"))
    monkeypatch.setattr(dm, "_get_preferred_farm_polygon", lambda oid: None)

    saved, failed, _ = dm.save_farm_risk_to_db(_total_risk_df(farm_id=""), OID_ANALYSIS)

    assert (saved, failed) == (1, 0)


def test_save_farm_risk_reports_unknown_farms(dm, risk_orm):
    saved, failed, errors = dm.save_farm_risk_to_db(_total_risk_df(farm_id=""), OID_ANALYSIS)

    assert (saved, failed) == (0, 1)
    assert errors[0]["error"] == "Farm no encontrado"


def test_save_farm_risk_ignores_invalid_object_ids(dm, risk_orm):
    # El farm_id del CSV no es un ObjectId: se ignora y se cae al camino por
    # sitcode, que tampoco encuentra la finca.
    saved, failed, errors = dm.save_farm_risk_to_db(
        _total_risk_df(farm_id="no-es-un-oid"), OID_ANALYSIS
    )

    assert (saved, failed) == (0, 1)
    assert errors[0]["error"] == "Farm no encontrado"


def test_save_farm_risk_reports_missing_analysis(dm, orm):
    Analysis = orm["Analysis"]
    Analysis.reset([])
    Analysis.objects = _analysis_manager(Analysis)

    saved, failed, errors = dm.save_farm_risk_to_db(_total_risk_df(), OID_ANALYSIS)

    assert (saved, failed) == (0, 0)
    assert "no encontrado" in errors[0]["error"]


def test_save_farm_risk_reports_analysis_query_errors(dm, orm):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    orm["Analysis"].objects = Manager()

    saved, failed, errors = dm.save_farm_risk_to_db(_total_risk_df(), OID_ANALYSIS)

    assert "Error obteniendo Analysis" in errors[0]["error"]


def test_save_farm_risk_records_save_failures(dm, risk_orm):
    risk_orm["FarmRisk"].save_error = RuntimeError("escritura rechazada")

    saved, failed, errors = dm.save_farm_risk_to_db(_total_risk_df(), OID_ANALYSIS)

    assert (saved, failed) == (0, 1)
    assert "escritura rechazada" in errors[0]["error"]


def test_save_farm_risk_without_orm(dm, no_orm):
    saved, failed, errors = dm.save_farm_risk_to_db(_total_risk_df(), OID_ANALYSIS)

    assert errors == [{"error": "ganabosques_orm no disponible"}]


# ===================== save_farm_risk_to_db_bulk =====================


def test_save_farm_risk_bulk_inserts_documents(dm, risk_orm, capsys):
    saved, failed, errors = dm.save_farm_risk_to_db_bulk(_total_risk_df(), OID_ANALYSIS)

    assert (saved, failed, errors) == (1, 0, [])
    assert "Usando farm_id y farm_poligons_id directamente" in capsys.readouterr().out
    assert dm.get_farmrisk_cache_for_analysis(OID_ANALYSIS)


def test_save_farm_risk_bulk_splits_chunks(dm, risk_orm):
    df = pd.concat(
        [_total_risk_df(id=f"S{i}", farm_id=f"507f1f77bcf86cd79943902{i}") for i in range(5)],
        ignore_index=True,
    )

    saved, failed, _ = dm.save_farm_risk_to_db_bulk(df, OID_ANALYSIS, chunk_size=2)

    assert (saved, failed) == (5, 0)


def test_save_farm_risk_bulk_reports_missing_analysis(dm, orm):
    Analysis = orm["Analysis"]
    Analysis.reset([])
    Analysis.objects = _analysis_manager(Analysis)

    saved, failed, errors = dm.save_farm_risk_to_db_bulk(_total_risk_df(), OID_ANALYSIS)

    assert "no encontrado" in errors[0]["error"]


def test_save_farm_risk_bulk_reports_analysis_errors(dm, orm):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    orm["Analysis"].objects = Manager()

    _, _, errors = dm.save_farm_risk_to_db_bulk(_total_risk_df(), OID_ANALYSIS)

    assert "Error validando analysis" in errors[0]["error"]


def test_save_farm_risk_bulk_falls_back_to_manual_resolution(dm, risk_orm, capsys):
    dm._farms_metadata = [{"sitcode": "S1", "mongo_id": OID_FARM, "farm_polygon_id": OID_POLYGON}]
    df = _total_risk_df().drop(columns=["farm_id", "farm_poligons_id"])

    saved, failed, _ = dm.save_farm_risk_to_db_bulk(df, OID_ANALYSIS)

    assert (saved, failed) == (1, 0)
    assert "resolviendo sitcodes manualmente" in capsys.readouterr().out


def test_save_farm_risk_bulk_reports_insert_errors(dm, risk_orm):
    risk_orm["FarmRisk"].insert_error = RuntimeError("duplicate key")

    saved, failed, errors = dm.save_farm_risk_to_db_bulk(_total_risk_df(), OID_ANALYSIS)

    assert (saved, failed) == (0, 1)
    assert "duplicados" in errors[0]["error"]


@pytest.mark.parametrize(
    "mensaje,esperado",
    [
        ("unique index violated", "duplicados"),
        ("validation failed", "validación"),
        ("otro problema", "Error de inserción"),
    ],
)
def test_bulk_insert_classifies_error_types(dm, risk_orm, mensaje, esperado):
    risk_orm["FarmRisk"].insert_error = RuntimeError(mensaje)

    _, _, errors = dm.save_farm_risk_to_db_bulk(_total_risk_df(), OID_ANALYSIS)

    assert esperado in errors[0]["error"]


def test_save_farm_risk_bulk_without_orm(dm, no_orm):
    _, _, errors = dm.save_farm_risk_to_db_bulk(_total_risk_df(), OID_ANALYSIS)

    assert errors == [{"error": "ganabosques_orm no disponible"}]


def test_bulk_insert_returns_empty_for_no_docs(dm):
    assert dm._bulk_insert_farm_risks([]) == (0, 0, [], {})


# ===================== _bulk_resolve_farms =====================


def test_bulk_resolve_farms_uses_metadata(dm, orm):
    dm._farms_metadata = [
        {"sitcode": "S1", "mongo_id": "f1", "farm_polygon_id": "p1"},
        {"sitcode": "S2", "mongo_id": "f2", "farm_polygon_id": None},
        {"sitcode": None, "mongo_id": "f3"},
    ]

    mapa = dm._bulk_resolve_farms(["S1", "S2"])

    assert mapa == {
        "S1": {"farm_id": "f1", "farm_polygon_id": "p1"},
        "S2": {"farm_id": "f2", "farm_polygon_id": ""},
    }


def test_bulk_resolve_farms_falls_back_to_database(dm, orm, monkeypatch):
    orm["Farm"]._store.append(_farm("f1", sit="S1"))
    monkeypatch.setattr(dm, "_build_preferred_polygon_map", lambda ids: {"f1": "p1"})

    mapa = dm._bulk_resolve_farms(["S1"])

    assert mapa == {"S1": {"farm_id": "f1", "farm_polygon_id": "p1"}}


def test_bulk_resolve_farms_from_db_without_orm(dm, no_orm):
    assert dm._bulk_resolve_farms_from_db(["S1"]) == {}


def test_bulk_resolve_farms_from_db_handles_errors(dm, orm):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    orm["Farm"].objects = Manager()

    assert dm._bulk_resolve_farms_from_db(["S1"]) == {}


def test_bulk_resolve_farms_from_db_without_matches(dm, orm):
    assert dm._bulk_resolve_farms_from_db(["S1"]) == {}


# ===================== _prepare_farm_risk_docs =====================


def test_prepare_farm_risk_docs_reports_missing_csv_ids(dm):
    df = _total_risk_df(farm_id="")

    docs, errors = dm._prepare_farm_risk_docs(df, OID_ANALYSIS, None, True)

    assert docs == []
    assert errors[0]["error"] == "farm_id vacío en CSV"


def test_prepare_farm_risk_docs_reports_missing_map_entry(dm):
    docs, errors = dm._prepare_farm_risk_docs(_total_risk_df(), OID_ANALYSIS, {}, False)

    assert errors[0]["error"] == "Farm no encontrado en mapeo"


def test_prepare_farm_risk_docs_reports_map_entry_without_farm_id(dm):
    mapa = {"S1": {"farm_id": "", "farm_polygon_id": ""}}

    docs, errors = dm._prepare_farm_risk_docs(_total_risk_df(), OID_ANALYSIS, mapa, False)

    assert errors[0]["error"] == "Farm sin mongo_id"


def test_prepare_farm_risk_docs_reports_invalid_object_ids(dm):
    docs, errors = dm._prepare_farm_risk_docs(
        _total_risk_df(farm_id="no-es-un-oid"), OID_ANALYSIS, None, True
    )

    assert docs == []
    assert errors[0]["error"] == "ObjectId inválido"


def test_prepare_farm_risk_docs_handles_null_polygon_id(dm, orm):
    docs, errors = dm._prepare_farm_risk_docs(
        _total_risk_df(farm_poligons_id="nan"), OID_ANALYSIS, None, True
    )

    assert errors == []
    assert docs[0]["farm_polygons_id"] is None


def test_prepare_farm_risk_docs_captures_unexpected_errors(dm):
    class FilaRota(dict):
        def get(self, key, default=None):
            if key == "deforested_ha":
                raise RuntimeError("fila corrupta")
            return super().get(key, default)

    class DfFalso:
        def iterrows(self):
            yield 0, FilaRota(
                {"id": "S1", "farm_id": OID_FARM, "farm_poligons_id": OID_POLYGON}
            )

    docs, errors = dm._prepare_farm_risk_docs(DfFalso(), OID_ANALYSIS, None, True)

    assert docs == []
    assert "Error preparando doc" in errors[0]["error"]


def test_bulk_insert_maps_plain_object_ids(dm, risk_orm, monkeypatch):
    """Cuando insert() devuelve ObjectIds en vez de documentos, igual se indexan."""

    def manager_insert(docs, load_bulk=False):
        return ["oid-generado"]

    risk_orm["FarmRisk"].objects.insert = manager_insert

    saved, failed, errors, refs = dm._bulk_insert_farm_risks(
        [
            {
                "farm_id": "f1",
                "analysis_id": OID_ANALYSIS,
                "farm_polygons_id": None,
                "external_farm_id": "FARM_ID_S1",
                "deforestation": {"ha": 1.0, "prop": 0.1},
                "protected": {"ha": 0.0, "prop": 0.0},
                "farming_in": {"ha": 0.0, "prop": 0.0},
                "farming_out": {"ha": 0.0, "prop": 0.0},
                "risk_direct": True,
                "risk_input": False,
                "risk_output": False,
            }
        ]
    )

    assert (saved, failed, errors) == (1, 0, [])
    assert refs == {"S1": "oid-generado"}


def test_bulk_insert_skips_docs_without_external_id(dm, risk_orm):
    saved, _, _, refs = dm._bulk_insert_farm_risks(
        [
            {
                "farm_id": "f1",
                "analysis_id": OID_ANALYSIS,
                "farm_polygons_id": None,
                "external_farm_id": "",
                "deforestation": {"ha": 0.0, "prop": 0.0},
                "protected": {"ha": 0.0, "prop": 0.0},
                "farming_in": {"ha": 0.0, "prop": 0.0},
                "farming_out": {"ha": 0.0, "prop": 0.0},
                "risk_direct": False,
                "risk_input": False,
                "risk_output": False,
            }
        ]
    )

    assert saved == 1
    assert refs == {}


# ===================== save_enterprise_risk_to_db =====================


def _enterprise_df(**overrides):
    fila = {
        "idpro": "E1",
        "id_farm": "S1",
        "typemove": "in",
        "enterprise_type_raw": "SLAUGHTERHOUSE",
    }
    fila.update(overrides)
    return pd.DataFrame([fila])


@pytest.fixture
def enterprise_orm(dm, risk_orm):
    from ganabosques_orm.enums.label import Label
    from ganabosques_orm.enums.typeenterprise import TypeEnterprise

    Enterprise = risk_orm["Enterprise"]
    Enterprise.reset(
        [
            Enterprise(
                id=OID_ENTERPRISE,
                ext_id=[Ext(None, "E1", label=Label.PRODUCTIONUNIT_ID)],
                type_enterprise=TypeEnterprise.SLAUGHTERHOUSE,
            )
        ]
    )
    Enterprise.objects = EnterpriseObjectsManager(Enterprise._store)

    EnterpriseRisk = risk_orm["EnterpriseRisk"]
    EnterpriseRisk.reset([])

    class Manager:
        def __call__(self, **filters):
            items = [
                er for er in EnterpriseRisk._store
                if str(er.enterprise_id) == str(filters.get("enterprise_id"))
                and str(er.analysis_id) == str(filters.get("analysis_id"))
            ]
            return _QuerySet(items)

    EnterpriseRisk.objects = Manager()
    return risk_orm


def test_save_enterprise_risk_creates_document(dm, enterprise_orm):
    dm._farmrisk_cache_by_analysis[OID_ANALYSIS] = {"S1": "fr1"}

    saved, failed, errors = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert (saved, failed, errors) == (1, 0, [])
    doc = enterprise_orm["EnterpriseRisk"]._store[0]
    assert doc.risk_input == ["fr1"]
    assert doc.risk_output == []


def test_save_enterprise_risk_uses_explicit_map(dm, enterprise_orm, capsys):
    saved, _, _ = dm.save_enterprise_risk_to_db(_enterprise_df(typemove="out"), OID_ANALYSIS, farm_risk_map={"S1": "fr9"}
    )

    assert saved == 1
    assert enterprise_orm["EnterpriseRisk"]._store[0].risk_output == ["fr9"]
    assert "Usando cache FarmRisk en memoria" in capsys.readouterr().out


def test_save_enterprise_risk_falls_back_to_database(dm, enterprise_orm, capsys):
    enterprise_orm["Farm"]._store.append(_farm(OID_FARM, sit="S1"))
    FarmRisk = enterprise_orm["FarmRisk"]
    fr = FarmRisk(id="fr_db", farm_id=OID_FARM, analysis_id=OID_ANALYSIS)
    FarmRisk.reset([fr])
    FarmRisk.objects = _farmrisk_manager(FarmRisk)

    saved, _, _ = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert saved == 1
    assert enterprise_orm["EnterpriseRisk"]._store[0].risk_input == [fr]
    assert "Cache FarmRisk vacío" in capsys.readouterr().out


def test_save_enterprise_risk_skips_farms_not_in_database(dm, enterprise_orm):
    saved, _, _ = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert saved == 1
    assert enterprise_orm["EnterpriseRisk"]._store[0].risk_input == []


def test_save_enterprise_risk_updates_without_duplicating(dm, enterprise_orm):
    EnterpriseRisk = enterprise_orm["EnterpriseRisk"]
    existente = EnterpriseRisk(
        id="er1", enterprise_id=OID_ENTERPRISE, analysis_id=OID_ANALYSIS,
        risk_input=["fr1"], risk_output=[],
    )
    EnterpriseRisk._store.append(existente)
    dm._farmrisk_cache_by_analysis[OID_ANALYSIS] = {"S1": "fr1"}

    saved, _, _ = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert saved == 1
    assert existente.risk_input == ["fr1"]


def test_save_enterprise_risk_appends_new_references(dm, enterprise_orm):
    EnterpriseRisk = enterprise_orm["EnterpriseRisk"]
    existente = EnterpriseRisk(
        id="er1", enterprise_id=OID_ENTERPRISE, analysis_id=OID_ANALYSIS,
        risk_input=["fr_previo"], risk_output=[],
    )
    EnterpriseRisk._store.append(existente)
    dm._farmrisk_cache_by_analysis[OID_ANALYSIS] = {"S1": "fr_nuevo"}

    dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert existente.risk_input == ["fr_previo", "fr_nuevo"]


def test_save_enterprise_risk_rejects_unknown_enterprise_type(dm, enterprise_orm):
    saved, failed, errors = dm.save_enterprise_risk_to_db(_enterprise_df(enterprise_type_raw="TIPO_RARO"), OID_ANALYSIS
    )

    assert (saved, failed) == (0, 1)
    assert "Tipo de empresa desconocido" in errors[0]["error"]


def test_save_enterprise_risk_accepts_lowercase_types(dm, enterprise_orm):
    saved, failed, _ = dm.save_enterprise_risk_to_db(_enterprise_df(enterprise_type_raw="slaughterhouse"), OID_ANALYSIS
    )

    assert (saved, failed) == (1, 0)


def test_save_enterprise_risk_falls_back_to_search_without_type(dm, enterprise_orm):
    from ganabosques_orm.enums.typeenterprise import TypeEnterprise

    # La empresa existe con otro tipo: se encuentra en la segunda búsqueda.
    enterprise_orm["Enterprise"]._store[0].type_enterprise = TypeEnterprise.CATTLE_FAIR

    saved, failed, _ = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert (saved, failed) == (1, 0)


def test_save_enterprise_risk_reports_unknown_enterprise(dm, enterprise_orm):
    enterprise_orm["Enterprise"].reset([])
    enterprise_orm["Enterprise"].objects = EnterpriseObjectsManager([])

    saved, failed, errors = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert (saved, failed) == (0, 1)
    assert "Enterprise no encontrado" in errors[0]["error"]


def test_save_enterprise_risk_uses_alternative_column_names(dm, enterprise_orm):
    df = pd.DataFrame(
        [{"enterprise_id": "E1", "id_farm": "S1", "typemove": "in", "type_enterprise": "SLAUGHTERHOUSE"}]
    )

    saved, failed, _ = dm.save_enterprise_risk_to_db(df, OID_ANALYSIS)

    assert (saved, failed) == (1, 0)


def test_save_enterprise_risk_records_save_failures(dm, enterprise_orm):
    enterprise_orm["EnterpriseRisk"].save_error = RuntimeError("escritura rechazada")

    saved, failed, errors = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert (saved, failed) == (0, 1)
    assert "escritura rechazada" in errors[0]["error"]


def test_save_enterprise_risk_reports_missing_analysis(dm, enterprise_orm):
    Analysis = enterprise_orm["Analysis"]
    Analysis.reset([])
    Analysis.objects = _analysis_manager(Analysis)

    _, _, errors = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert "no encontrado" in errors[0]["error"]


def test_save_enterprise_risk_reports_analysis_errors(dm, enterprise_orm):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    enterprise_orm["Analysis"].objects = Manager()

    _, _, errors = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert "Error obteniendo Analysis" in errors[0]["error"]


def test_save_enterprise_risk_without_orm(dm, no_orm):
    _, _, errors = dm.save_enterprise_risk_to_db(_enterprise_df(), OID_ANALYSIS)

    assert errors == [{"error": "ganabosques_orm no disponible"}]


# ===================== capas de referencia y Analysis =====================


def test_get_latest_reference_layers(dm, orm):
    ProtectedAreas, FarmingAreas = orm["ProtectedAreas"], orm["FarmingAreas"]
    ProtectedAreas.reset(
        [ProtectedAreas(id="pa1", name="PNN 2023"), ProtectedAreas(id="pa2", name="PNN 2024")]
    )
    FarmingAreas.reset([FarmingAreas(id="fa1", name="UPRA 2024")])

    class Manager:
        def __init__(self, store):
            self.store = store

        def __call__(self, **filters):
            return _QuerySet(list(self.store))

    ProtectedAreas.objects = Manager(ProtectedAreas._store)
    FarmingAreas.objects = Manager(FarmingAreas._store)

    assert dm._get_latest_protected_areas().name == "PNN 2023"
    assert dm._get_latest_farming_areas().name == "UPRA 2024"


def test_get_latest_reference_layers_return_none_when_empty(dm, orm):
    class Manager:
        def __call__(self, **filters):
            return _QuerySet([])

    orm["ProtectedAreas"].objects = Manager()
    orm["FarmingAreas"].objects = Manager()

    assert dm._get_latest_protected_areas() is None
    assert dm._get_latest_farming_areas() is None


def test_get_latest_reference_layers_return_none_on_error(dm, orm):
    class Manager:
        def __call__(self, **filters):
            raise RuntimeError("mongo caído")

    orm["ProtectedAreas"].objects = Manager()
    orm["FarmingAreas"].objects = Manager()

    assert dm._get_latest_protected_areas() is None
    assert dm._get_latest_farming_areas() is None


def test_get_latest_reference_layers_without_orm(dm, no_orm):
    assert dm._get_latest_protected_areas() is None
    assert dm._get_latest_farming_areas() is None


@pytest.fixture
def analysis_orm(dm, orm, monkeypatch):
    Deforestation, Analysis = orm["Deforestation"], orm["Analysis"]
    Deforestation.reset([Deforestation(id="defo1", name="capa")])

    class DefoManager:
        def __call__(self, **filters):
            items = [d for d in Deforestation._store if str(d.id) == str(filters.get("id"))]
            return _QuerySet(items)

    class AnalysisManager:
        def __call__(self, **filters):
            items = [
                a for a in Analysis._store
                if str(getattr(a, "deforestation_id", None)) == str(filters.get("deforestation_id"))
                and getattr(a, "value_chain", None) == filters.get("value_chain")
            ]
            return _QuerySet(items)

    Deforestation.objects = DefoManager()
    Analysis.reset([])
    Analysis.objects = AnalysisManager()

    monkeypatch.setattr(dm, "_get_latest_protected_areas", lambda: SimpleNamespace(id="pa1", name="PNN"))
    monkeypatch.setattr(dm, "_get_latest_farming_areas", lambda: SimpleNamespace(id="fa1", name="UPRA"))
    return orm


def test_create_or_get_analysis_creates_new(dm, analysis_orm, capsys):
    from ganabosques_orm.enums.valuechain import ValueChain

    analysis_id, error = dm.create_or_get_analysis("defo1", value_chain="livestock")

    assert error is None
    doc = analysis_orm["Analysis"]._store[0]
    assert str(doc.id) == analysis_id
    assert doc.value_chain is ValueChain.LIVESTOCK
    assert doc.protected_areas_id.name == "PNN"
    salida = capsys.readouterr().out
    assert "Análisis creado" in salida


def test_create_or_get_analysis_returns_existing(dm, analysis_orm, capsys):
    Analysis = analysis_orm["Analysis"]
    Analysis._store.append(
        Analysis(id="an_previo", deforestation_id="defo1", value_chain="livestock")
    )

    analysis_id, error = dm.create_or_get_analysis("defo1")

    assert (analysis_id, error) == ("an_previo", None)
    assert "Usando análisis existente" in capsys.readouterr().out


def test_create_or_get_analysis_defaults_unknown_value_chain(dm, analysis_orm):
    from ganabosques_orm.enums.valuechain import ValueChain

    analysis_id, error = dm.create_or_get_analysis("defo1", value_chain="inexistente")

    assert error is None
    assert analysis_orm["Analysis"]._store[0].value_chain is ValueChain.LIVESTOCK


def test_create_or_get_analysis_warns_about_missing_reference_layers(dm, analysis_orm, monkeypatch, capsys):
    monkeypatch.setattr(dm, "_get_latest_protected_areas", lambda: None)
    monkeypatch.setattr(dm, "_get_latest_farming_areas", lambda: None)

    dm.create_or_get_analysis("defo1")

    salida = capsys.readouterr().out
    assert "No se encontró capa de áreas protegidas" in salida
    assert "No se encontró capa de frontera agrícola" in salida


def test_create_or_get_analysis_accepts_user_id(dm, analysis_orm):
    dm.create_or_get_analysis("defo1", user_id="u1")

    assert analysis_orm["Analysis"]._store[0].user_id == "u1"


def test_create_or_get_analysis_reports_missing_deforestation(dm, analysis_orm):
    analysis_id, error = dm.create_or_get_analysis("inexistente")

    assert analysis_id is None
    assert "no encontrado" in error


def test_create_or_get_analysis_reports_errors(dm, analysis_orm, monkeypatch):
    monkeypatch.setattr(
        data_manager, "ObjectId", lambda v=None: (_ for _ in ()).throw(ValueError("oid malo"))
    )

    analysis_id, error = dm.create_or_get_analysis("defo1")

    assert analysis_id is None
    assert "Error creando análisis" in error


def test_create_or_get_analysis_without_orm(dm, no_orm):
    assert dm.create_or_get_analysis("defo1") == (None, "ganabosques_orm no disponible")


# ===================== ramas restantes =====================


def test_get_preferred_farm_polygon_falls_back_to_disabled(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    deshabilitado = FarmPolygons(id="p_off", farm_id="f1", log=SimpleNamespace(enable=False))
    FarmPolygons.reset([deshabilitado])

    class Manager:
        def __call__(self, **filters):
            items = [p for p in FarmPolygons._store if p.farm_id == filters["farm_id"]]
            if filters.get("log__enable"):
                items = [p for p in items if p.log.enable]
            return _QuerySet(items)

    FarmPolygons.objects = Manager()

    # Sin ninguno habilitado se acepta cualquier polígono disponible.
    assert dm._get_preferred_farm_polygon("f1") is deshabilitado


def test_prepare_geojsons_force_download_counts_documents_without_geojson(dm, orm):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([FarmPolygons(id="p1", farm_id=Ref(OID_FARM), geojson=None)])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)

    total = dm.prepare_geojsons(
        [{"mongo_id": OID_FARM, "sitcode": "A1"}], force_download=True
    )

    assert total == 0


def test_prepare_geojsons_counts_failures_in_download_batch(dm, orm):
    """Un doc sin geojson dentro del lote de faltantes se cuenta como fallo."""
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([FarmPolygons(id="p1", farm_id=Ref(OID_FARM), geojson=None)])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)

    total = dm.prepare_geojsons(
        [
            {"mongo_id": OID_FARM, "sitcode": "A1"},
            {"mongo_id": None, "sitcode": "A2"},
        ]
    )

    assert total == 0


def test_prepare_geojsons_force_download_writes_with_mongo_id(dm, orm, monkeypatch):
    """Si el helper recibe sitcode vacío, el archivo toma el nombre del mongo_id."""
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([_polygon_doc(FarmPolygons, OID_FARM)])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)

    # _download_geojson comparte la misma lógica de nombrado.
    ruta = dm._download_geojson(OID_FARM, None)

    assert Path(ruta).name == f"{OID_FARM}.geojson"


def test_download_geojson_reports_write_errors(dm, orm, monkeypatch):
    FarmPolygons = orm["FarmPolygons"]
    FarmPolygons.reset([_polygon_doc(FarmPolygons, OID_FARM)])
    FarmPolygons.objects = _polygons_manager(FarmPolygons)
    monkeypatch.setattr(
        data_manager.json, "dump", lambda *a, **k: (_ for _ in ()).throw(OSError("disco lleno"))
    )

    assert dm._download_geojson(OID_FARM, "A1") is None


def test_get_geometry_from_cache_shortcut(dm):
    geom = box(0, 0, 10, 10)
    dm._geometries_cache["A1"] = geom

    assert dm.get_geometry("A1") is geom


def test_download_movements_handles_plain_string_sources(dm, orm, movement_module):
    """Cuando ext.source es un string plano (no un Enum) también se interpreta."""
    origen = SimpleNamespace(ext_id=[Ext("SIT_CODE", "S1")])
    destino = SimpleNamespace(
        ext_id=[Ext("SIT_CODE", "S2"), Ext("PRODUCER_ID", "P2")]
    )
    movement_module.reset([_mov(origen=origen, destino=destino)])

    ruta, error = dm.download_movements_to_csv(2024)

    assert error is None
    df = pd.read_csv(ruta, dtype=str)
    assert df.iloc[0]["SIT_CODE_ORIGEN"] == "S1"
    assert df.iloc[0]["PRODUCER_ID_DESTINO"] == "P2"


def test_download_movements_handles_missing_dates_and_types(dm, orm, movement_module):
    mov = _mov()
    mov.date = None
    mov.type_origin = None
    mov.type_destination = None
    movement_module.reset([mov])

    ruta, error = dm.download_movements_to_csv(2024)

    df = pd.read_csv(ruta, dtype=str, keep_default_na=False)
    assert df.iloc[0]["DATE"] == ""
    assert df.iloc[0]["TIPO_ORIGEN"] == ""


def test_safe_converters_handle_unparseable_values(dm, risk_orm):
    """'abc' no es convertible: cae en el valor por defecto sin romper la fila."""
    saved, failed, _ = dm.save_farm_risk_to_db(
        _total_risk_df(deforested_ha="abc", direct_alert=True), OID_ANALYSIS
    )

    assert (saved, failed) == (1, 0)
    doc = risk_orm["FarmRisk"]._store[0]
    assert doc.deforestation.ha == 0.0
    # Un booleano nativo se respeta tal cual.
    assert doc.risk_direct is True


def test_safe_converters_handle_none_values(dm, risk_orm):
    saved, failed, _ = dm.save_farm_risk_to_db(
        _total_risk_df(deforested_prop=None, indirect_alert_in=None), OID_ANALYSIS
    )

    assert (saved, failed) == (1, 0)
    doc = risk_orm["FarmRisk"]._store[0]
    assert doc.deforestation.prop == 0.0
    assert doc.risk_input is False


def test_bulk_safe_converters_handle_unparseable_values(dm, risk_orm):
    docs, errors = dm._prepare_farm_risk_docs(
        _total_risk_df(deforested_ha="abc", deforested_prop=None, direct_alert=True),
        OID_ANALYSIS,
        None,
        True,
    )

    assert errors == []
    assert docs[0]["deforestation"] == {"ha": 0.0, "prop": 0.0}
    assert docs[0]["risk_direct"] is True


def test_save_enterprise_risk_appends_new_output_references(dm, enterprise_orm):
    EnterpriseRisk = enterprise_orm["EnterpriseRisk"]
    existente = EnterpriseRisk(
        id="er1", enterprise_id=OID_ENTERPRISE, analysis_id=OID_ANALYSIS,
        risk_input=[], risk_output=["fr_previo"],
    )
    EnterpriseRisk._store.append(existente)
    dm._farmrisk_cache_by_analysis[OID_ANALYSIS] = {"S1": "fr_nuevo"}

    dm.save_enterprise_risk_to_db(_enterprise_df(typemove="out"), OID_ANALYSIS)

    assert existente.risk_output == ["fr_previo", "fr_nuevo"]


def test_save_enterprise_risk_skips_duplicate_output_references(dm, enterprise_orm):
    EnterpriseRisk = enterprise_orm["EnterpriseRisk"]
    existente = EnterpriseRisk(
        id="er1", enterprise_id=OID_ENTERPRISE, analysis_id=OID_ANALYSIS,
        risk_input=[], risk_output=["fr1"],
    )
    EnterpriseRisk._store.append(existente)
    dm._farmrisk_cache_by_analysis[OID_ANALYSIS] = {"S1": "fr1"}

    dm.save_enterprise_risk_to_db(_enterprise_df(typemove="out"), OID_ANALYSIS)

    assert existente.risk_output == ["fr1"]
