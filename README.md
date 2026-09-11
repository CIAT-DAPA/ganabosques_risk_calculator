# Risk Calculator

![GitHub release (latest by date)](https://img.shields.io/github/v/release/CIAT-DAPA/ganabosques_risk_calculator) ![](https://img.shields.io/github/v/tag/CIAT-DAPA/ganabosques_risk_calculator)

The Risk Calculator container operates as an ETL (Extract, Transform, and Load) process that quantifies deforestation risk for livestock and cacao farms in Colombia. It consumes the farm polygons produced by the [Farm Data Processor](https://github.com/CIAT-DAPA/ganabosques_farm_data_processor), the deforestation rasters published in GeoServer, and the cattle movement records stored in MongoDB, and it produces a consolidated risk profile at four levels: farm (`FarmRisk`), enterprise (`EnterpriseRisk`), supplier-linked enterprise, and administrative unit (`Adm3Risk`).

The risk model is organized in complementary layers:

- **Direct risk** — the farm polygon is intersected against the deforestation raster of each period to obtain deforested hectares, the deforested proportion of the polygon, and the boolean direct alert.
- **Indirect risk** — cattle movements are traced in both directions, so a farm inherits risk when it receives animals from (`indirect_alert_in`) or ships animals to (`indirect_alert_out`) a farm that carries a direct alert.
- **Spatial metrics** — each polygon is measured against the UPRA agricultural frontier and the National Natural Parks layer, producing `farming_in`, `farming_out`, and `protected` areas and proportions used as context for the risk score.
- **Aggregated risk** — farm-level results are rolled up to enterprises (through movements or through the `Suppliers` collection) and to ADM3 administrative units.

The scoring rules themselves live in the shared [ganabosques_risk_package](https://github.com/CIAT-DAPA/ganabosques_risk_package), while database access is handled through [ganabosques_orm](https://github.com/CIAT-DAPA/ganabosques_orm). This repository is the orchestration layer: it resolves which periods actually exist in the database, caches the geospatial inputs locally, runs the calculations (sequentially or in parallel), writes reviewable CSVs, and persists the approved results back to MongoDB.


## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/CIAT-DAPA/ganabosques_risk_calculator.git
```

2. Create a virtual environment:
```bash
python -m venv env
```

3. Activate the virtual environment:
```bash
env\Scripts\activate  # Windows
source env/bin/activate  # Linux/Mac
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

> The project targets **Python 3.10** (the version used by the CI workflow). `requirements.txt` installs `ganabosques_orm` and `ganabosques_risk_package` directly from GitHub, so `git` must be available on the machine.

5. Create the configuration file from the template and fill in your credentials:
```bash
cp src/.env.example src/.env   # Linux/Mac
copy src\.env.example src\.env # Windows
```


## 🧱 Project Structure

The project follows a modular structure where each module implements one stage of the risk pipeline, and `main.py` orchestrates them:

```bash
├── src/
│   ├── main.py                  # Pipeline entry point: CLI, period resolution, step orchestration
│   ├── config.py                # Loads .env, exposes the config dict, RiskLevel / classify_risk
│   ├── data_manager.py          # Data layer: farm metadata, GeoJSON & raster cache, MongoDB persistence
│   ├── direct_alert.py          # Step 1: farm polygon × deforestation raster intersection
│   ├── indirect_alert.py        # Step 2: indirect alerts from cattle movements (+ enterprise alerts)
│   ├── spatial_metrics.py       # Step 3: agricultural frontier and protected area metrics
│   ├── enterprise_alert.py      # Step 4: standalone enterprise alerts from movements
│   ├── total_alert.py           # Step 5: consolidation of direct + indirect + metrics
│   ├── adm3_alert.py            # Steps 7-8: risk aggregated by administrative unit (ADM3)
│   ├── supplier_alert.py        # Step 9: enterprise risk derived from the Suppliers collection
│   ├── parallel_processor.py    # Multiprocessing support for direct alerts and spatial metrics
│   ├── error_logger.py          # Per-row error logging to CSV for post-run review
│   ├── utils.py                 # Shared helpers (areas, ID normalization, period parsing, logging)
│   ├── .env.example             # Template of the environment configuration
│   └── .env                     # Environment config (MongoDB, GeoServer, workspace) — not versioned
├── tests/                       # Unit tests for each pipeline stage
├── .github/workflows/           # CI: run tests on `stage` and merge into `main`
├── pytest.ini                   # Pytest config (adds src/ to the import path)
├── requirements.txt             # Python dependencies
└── README.md
```

### 🔌 External dependencies

| Dependency | Role |
|------------|------|
| MongoDB | Farms, farm polygons, deforestation layer catalog, movements, enterprises, suppliers, and all risk collections |
| GeoServer (WCS) | Source of the deforestation rasters, downloaded on demand and cached locally |
| GeoServer (WFS) | Source of the reference layers `administrative:upra_boundaries` and `administrative:pnn_areas` |
| `ganabosques_orm` | Document definitions and enums (`DeforestationSource`, `DeforestationType`, `ValueChain`, `Label`) |
| `ganabosques_risk_package` | Risk calculation rules shared with the rest of the Ganabosques platform |


## 🚀 How to Run the Pipeline

All commands are executed from the `src/` folder:

```bash
cd src
python main.py -s smbyc -pt annual -y 2017-2024
```

Before running, `main.py` queries MongoDB for the deforestation layers that match `--source` and `--period-type`, and **only the periods that actually exist in the database are processed**. If the requested range covers periods that are not registered, the pipeline warns you, lists what is available, and continues with the intersection.

### 🧭 Available Steps

| Step | Description                                                                 |
|------|-----------------------------------------------------------------------------|
| 1    | Direct alerts (farm polygon intersected with the deforestation raster)      |
| 2    | Indirect alerts from cattle movements (also emits the enterprise alerts)    |
| 3    | Spatial metrics (agricultural frontier and protected areas)                 |
| 4    | Standalone enterprise alerts (only needed if step 2 was not executed)       |
| 5    | Total risk consolidation (direct + indirect + metrics)                      |
| 6    | Save `Analysis`, `FarmRisk` and `EnterpriseRisk` to MongoDB                 |
| 7    | Calculate ADM3 risk (writes CSVs for review, nothing is saved yet)          |
| 8    | Save ADM3 risk to MongoDB (reads the CSVs produced by step 7)               |
| 9    | Enterprise risk from suppliers, for enterprises without cattle movements    |

Steps 7 and 8 are deliberately split so the aggregated municipal figures can be reviewed as CSV before they reach the database. Step 4 is skipped automatically when step 2 is part of the same run, because step 2 already produces the enterprise alerts in the same pass.

### 🧾 Arguments

```
--source (-s)            Deforestation source. Default and only current value: smbyc
--period-type (-pt)      Required. Deforestation type: annual, cumulative, nad, atd
--years (-y)             Required. Years to process: '2024' or '2010-2024'
--step (-p)              Steps to execute: '1', '1-3', '1-3 5', 'all'. Omit to run 1, 2, 3, 5, 6
--empresa (-e)           Enterprise name or external code (required by step 9)
--value-chain (-vc)      Value chain: livestock (default) or cacao
--continue-on-error      Keep going when a stage fails instead of stopping the pipeline
--farm-limit N           Process at most N farms (testing)
--parallel               Enable multiprocessing for direct alerts and spatial metrics
--workers (-w) N         Number of parallel workers (default: number of CPUs)
--precise-area           Super-sample pixels for a more accurate deforested area (5×5 by default)
--pixel-divisions N      Sub-pixel divisions per pixel; only applies with --precise-area
--offline                Use only local files, without connecting to MongoDB
--refresh-data           Ignore the metadata cache and re-download the GeoJSONs
--bulk-insert            Save to MongoDB with bulk insert (faster, INSERT only, no UPDATE)
--chunk-size N           Chunk size for bulk insert (default: 1000)
--dry-run                Run validations and show configuration without persisting results
```

`--period-type` selects both the raster family and the period format: `annual` and `cumulative` use `YYYY-YYYY` (for example `2013-2014`), while the quarterly alerts `nad` and `atd` use `YYYYQQ` (for example `202203`).

Running `-p` with no value prints the list of steps and exits, which is a quick way to check the numbering.

### 📌 Run the default pipeline (steps 1, 2, 3, 5 and 6):

```bash
python main.py -s smbyc -pt annual -y 2017-2024
```

### 🛠 Run selected steps:

```bash
python main.py -s smbyc -pt nad -y 2024 -p 1        # only direct alerts
python main.py -s smbyc -pt nad -y 2024 -p 1-3      # steps 1 to 3
python main.py -s smbyc -pt nad -y 2024 -p 1-3 5    # steps 1, 2, 3 and 5
```

### 🔁 Calculate and review ADM3 risk:

```bash
python main.py -s smbyc -pt annual -y 2010-2024 -p 7   # generates the CSVs
python main.py -s smbyc -pt annual -y 2010-2024 -p 8   # saves the reviewed CSVs to MongoDB
```

### 🏭 Calculate supplier-based enterprise risk:

```bash
python main.py -s smbyc -pt annual -y 2020-2024 -p 9 -e "Enterprise name"
```

### ⚡ Performance and testing options:

```bash
python main.py -s smbyc -pt nad -y 2024 -p 1 --farm-limit 10          # quick smoke run
python main.py -s smbyc -pt nad -y 2024 --parallel -w 8               # parallel execution
python main.py -s smbyc -pt nad -y 2024 -p 1 --precise-area           # sub-pixel accuracy
python main.py -s smbyc -pt nad -y 2024 -p 6 --bulk-insert            # fast database load
```

---

### 📖 Help

```bash
python main.py -h
```

### ✅ Prerequisite validation

Each selected step is validated before any calculation starts, and the pipeline reports exactly what is missing:

| Step | Requirements |
|------|--------------|
| 1 | `DataManager` available and farm metadata or local GeoJSONs; in offline mode, a folder containing `.geojson` files |
| 2 | `movements/movement_data_base_YYYY.csv` files, plus the direct alerts of step 1 |
| 3 | `DataManager` and farm metadata or local GeoJSONs |
| 4 | Movement files and the direct alerts of step 1 |
| 5 | Direct alerts of step 1; spatial metrics are optional but recommended |

Use `--continue-on-error` to downgrade these checks to warnings.

> **Cattle movements** are read from `WORKSPACE/alertas/movements/movement_data_base_YYYY.csv` and must be present before running steps 2 or 4. `DataManager.download_movements_to_csv(year)` exports them from the `Movement` collection in the expected format.


## 🧪 Running Tests

To run all unit tests:

```bash
pytest tests/
```

`pytest.ini` adds `src/` to the import path, so the tests import the pipeline modules directly and no additional configuration is needed. The suite currently holds **46 tests** and covers:

- ID normalization, period parsing, and step parsing (`test_utils.py`, `test_core_logic.py`, `test_main.py`)
- Workspace creation, GeoJSON sanitization, and raster download in `DataManager` (`test_data_manager.py`)
- Raster and vector handling in direct alerts (`test_direct_alert.py`, `test_extended_logic.py`)
- Indirect alerts, enterprise alerts, and supplier risk (`test_indirect_alert.py`, `test_enterprise_alert.py`, `test_supplier_alert.py`)
- Total risk consolidation and spatial metrics (`test_total_alert.py`, `test_spatial_metrics.py`)
- ADM3 aggregation, parallel processing, and error logging (`test_adm3_alert.py`, `test_parallel_processor.py`, `test_error_logger.py`)

The tests do not require MongoDB or GeoServer: external services are replaced with fixtures and monkeypatching.


## 🛠️ Environment Variables

Set these values in `src/.env` or in the system environment. `MONGO_URI` and `MONGO_DB_NAME` are mandatory — the pipeline fails at start-up if they are missing — and everything else has a default:

```bash
# ============================================
# MongoDB configuration (required)
# ============================================
MONGO_URI=mongodb://user:password@localhost:27017
MONGO_DB_NAME=ganabosques

# ============================================
# GeoServer configuration
# ============================================
URL_GEO=http://your-geoserver-url:8080/geoserver
GEO_USER=admin
GEO_PWD=geoserver

# ============================================
# Workspace (local data cache)
# ============================================
# Rasters, GeoJSONs, reference layers and results are stored here
WORKSPACE=D:/ganabosques/data/

# ============================================
# Processing configuration
# ============================================
# Projected CRS in meters used for every area computation
CRS_METROS=EPSG:3116
# Raster value that represents deforestation
DEFOREST_VALUE=2
# Records per batch when writing the alert CSVs
BATCH_SIZE=1000
# Optional slice of the farm file list, e.g. 1:5000
FARM_FILE_RANGE=
# Default number of workers for parallel tasks
TASK_MAX_WORKERS=8

# ============================================
# Logging
# ============================================
LOG_LEVEL=WARNING
LOG_FILE=risk_analysis.log
```

Every area is computed in `CRS_METROS`, so the reference layers and rasters are reprojected to that CRS before intersecting; changing this value changes the units of every hectare column.

`config.py` also accepts optional legacy paths (`FOLDER_GEOJSONS`, `RASTER_DEFOREST`, `SHP_PROTECTED`, `FARMING_FRONTIER_SHP`, `OUTPUT_CSV`, and related variables) that are used as a fallback when `DataManager` cannot manage the workspace, mainly to keep older offline runs working.


## 📂 Outputs

Everything is written under the folder configured in `WORKSPACE`, inside an `alertas/` subdirectory managed by `DataManager`:

```bash
WORKSPACE/alertas/
├── farms/
│   └── geojsons/                       # Farm polygons cached by sit_code or Mongo id
├── rasters/
│   └── {source}/{period_type}/         # Deforestation rasters downloaded from GeoServer (WCS)
├── reference_layers/
│   ├── upra_boundaries/                # Agricultural frontier (WFS → GeoPackage)
│   └── pnn_areas/                      # National Natural Parks (WFS → GeoPackage)
├── movements/
│   └── movement_data_base_{YYYY}.csv   # Cattle movements per year (pipeline input)
├── metrics/
│   └── spatial_metrics.csv             # Frontier and protected area metrics per farm
├── cache/                              # Intermediate caches (suppliers, mappings)
├── logs/
│   └── errors_{source}_{type}_*.csv    # Per-row errors collected during the run
└── results/
    └── {source}/{period_type}/
        ├── direct_alerts/{source}_direct_alert_{period_type}_{period}.csv
        ├── indirect_alerts/{source}_indirect_alert_{period_type}_{period}.csv
        ├── enterprise_alerts/{source}_enterprise_alert_{period_type}_{period}.csv
        ├── total_risk/{source}_total_risk_{period_type}_{period}.csv
        └── adm3_risk/{source}_adm3_risk_{period_type}_{period}.csv
```

Results are partitioned by source and deforestation type, so `annual`, `cumulative`, `nad`, and `atd` runs coexist without overwriting each other.

### 📄 Main output columns

| File | Columns |
|------|---------|
| `direct_alert` | `id`, `intersect_deforestation`, `deforested_ha`, `deforested_prop`, `direct_alert` |
| `indirect_alert` | `period`, `year`, `quarter`, `id`, `n_total_mov`, `n_in`, `n_out`, `n_indirect_in`, `n_indirect_out`, `indirect_alert_in`, `indirect_alert_out` |
| `enterprise_alert` | `idpro`, `id_farm`, `typemove` (`in`/`out`), `period`, `year`, `source`, `farm_has_direct_alert`, `enterprise_name`, `enterprise_type` |
| `spatial_metrics` | `id`, `total_ha`, `farming_in_ha`, `farming_in_prop`, `farming_out_ha`, `farming_out_prop`, `protected_ha`, `protected_prop` |
| `total_risk` | The direct, indirect, and metric columns consolidated per farm, plus `farm_id` and `farm_poligons_id` for the MongoDB link |
| `adm3_risk` | `adm3_id`, `analysis_id`, `def_ha`, `risk_total`, `farm_amount`, `farm_amount_total` |

### 🗄️ Data written to MongoDB

Steps 6, 8, and 9 persist the reviewed results through `ganabosques_orm`:

| Collection | Written by | Content |
|------------|-----------|---------|
| `Analysis` | Step 6 | One analysis per deforestation layer and value chain, created or reused as needed |
| `FarmRisk` | Step 6 | Per-farm risk read from the `total_risk` CSVs |
| `EnterpriseRisk` | Steps 6 and 9 | Enterprise risk from movements (step 6) or from suppliers (step 9) |
| `Adm3Risk` | Step 8 | Risk aggregated by administrative unit, read from the `adm3_risk` CSVs |

By default the load is granular and uses upsert, so re-running a period updates the existing documents. `--bulk-insert` is considerably faster but performs inserts only, so previous data for the period must be removed first to avoid duplicate-key errors.

### 📃 Logging

Two complementary logs are produced:

- The file configured in `LOG_FILE` (`risk_analysis.log` by default) with the execution trace at the `LOG_LEVEL` verbosity.
- A CSV under `WORKSPACE/alertas/logs/` where every failed row is recorded with its `timestamp`, `stage`, `farm_id`, `period`, `error_type`, and `error_message`, so partial failures can be reviewed and reprocessed after the run. A summary is printed at the end of the pipeline.


## 👥 Contributors

This project has had the following people who made contributions through commits to the repository:

- [bmora-0110](https://github.com/bmora-0110)
- [stevensotelo](https://github.com/stevensotelo)
- [victor-993](https://github.com/victor-993)