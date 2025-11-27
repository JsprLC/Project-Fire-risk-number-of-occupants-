# Probabilistic modeling for fire risk

A Bayesian Network-based system for assessing fire-related fatality risks in residential buildings.

## Overview

This project uses 3D building models (LoD2 CityGML) to:
1. Estimate number of occupants in residential buildings
2. Calculate probability distributions of fire-related fatalities using Bayesian Networks
3. Visualize risk on interactive maps

## Dataset

**Source:** [Bayern Open Geodata - LoD2 3D Building Models](https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=lod2)

**Data File:** `Data&Maps/001test.geojson`

Small subset of 3D building models from Munich, Germany with building volumes extracted from CityGML files.

## Installation

```bash
pip install numpy pandas geopandas scipy pgmpy folium matplotlib jupyter
```

## Quick Start

### Step 1: Preprocess Data & Estimate Occupancy

```bash
jupyter notebook 01GIS_data.ipynb
```

- Loads building data from `Data&Maps/001test.geojson`
- Estimates number of households and occupants per building
- Uses functions from `functions_occupants.py` and `functions_extract_gdf.py`

**Output:** Preprocessed building data with occupancy estimates

### Step 2: Run Bayesian Network Analysis

```bash
jupyter notebook 02BNs.ipynb
```

- Builds Bayesian Network for each building
- Based on building volume, building type, statistical distribution of probability of deaths and number of occupants
- Performs inference to get probability distribution of expected deaths
- Uses functions from `functions_BN.py`

**Output:** Building-level risk estimates

### Step 3: Visualize Results

```bash
python Mapping/Final_visualize_risk.py
```

See `Mapping/START_HERE.md` for detailed instructions.

**Output:** Interactive HTML map showing spatial distribution of fire risk
**Map Link:** https://jsprlc.github.io/Project-Fire-risk-number-of-occupants-/DataMaps/building_risk_FINAL.html

## Project Structure

```
├── Data&Maps/
    ├── 001test.geojson          # Input data
    ├── building_risk_FINAL.html # Output interactive map
    └── building_risk_static_maps.png
├── 01GIS_data.ipynb             # Step 1: Occupancy estimation
├── 02BNs.ipynb                  # Step 2: Risk analysis
├── functions_occupants.py       # Occupancy calculation functions
├── functions_extract_gdf.py     # GeoData extraction utilities
├── functions_BN.py              # Bayesian Network functions
└── Mapping/
    ├── Final_visualize_risk.py  # Visualization script
    └── START_HERE.md            # Visualization guide
```

## Key Functions

### functions_occupants.py
- `Building` dataclass - Store building attributes
- `av_storey_h_and_h_area_building()` - Calculate heated area
- `heated_area_per_household()` - Distribute area among households
- `calculate_building_occupants()` - Estimate total occupants

### functions_extract_gdf.py
- `extract_buildings_from_geodataframe()` - Extract buildings data from geodataframe

### functions_BN.py
- `BuildingRiskBayesianNetwork` class - Build and run Bayesian Network for a single building
- `get_expected_deaths_distribution()` - Get risk distribution

## License

[]

## Contact

[]
