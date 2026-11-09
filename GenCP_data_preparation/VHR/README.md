# VHR OSM Styled Dataset Preparation

This repository was developed within the GenCP project to prepare paired VHR raster tiles and corresponding styled OSM mask tiles for Generative AI-based image processing workflows.

The workflow is designed to be managed from:

```text
VHR_OSM_STYLED_DATASET_PREP.ipynb
```

The Python scripts provide OSM downloading, OSM attribute formatting, shapefile merging, QGIS-based styled mask rendering, and optional tile cleanup. The styled rendering step requires a local QGIS installation from [QGIS Download](https://qgis.org/download/) and is executed through `run.bat`, which loads the QGIS Python environment before calling `osm_style_processor.py`.

---

## Repository contents

```text
GenCP_data_preparation/VHR/
├── batch_runner.py
├── osm_processor.py
├── batch_formatter.py
├── osm_formatter_engine.py
├── merge_shp.py
├── osm_style_processor.py
├── cleanup_tiles.py
├── run.bat
├── requirements.txt
├── VHR_OSM_STYLED_DATASET_PREP.ipynb
├── raster          # Input VHR raster data folder, e.g. GeoTIFF images
└── Styles          # QML style files used for rendering OSM masks


Styles/
├── gis_osm_buildings_a_free_1.qml
├── gis_osm_landuse_a_free_1.qml
├── gis_osm_natural_a_free_1.qml
├── gis_osm_natural_free_1.qml
├── gis_osm_places_free_1.qml
├── gis_osm_pofw_a_free_1.qml
├── gis_osm_pofw_free_1.qml
├── gis_osm_pois_a_free_1.qml
├── gis_osm_pois_free_1.qml
├── gis_osm_railways_free_1.qml
├── gis_osm_roads_free_1.qml
├── gis_osm_traffic_a_free_1.qml
├── gis_osm_traffic_free_1.qml
├── gis_osm_transport_a_free_1.qml
├── gis_osm_transport_free_1.qml
├── gis_osm_water_a_free_1.qml
└── gis_osm_waterways_free_1.qml
```

---

### Installation

A new environment can be created as follows:

```bat
conda create -n vhr_osm python=3.10 -y
conda activate vhr_osm
conda install -c conda-forge --file requirements.txt
python -m ipykernel install --user --name vhr_osm --display-name "Python (vhr_osm)"
```


## QGIS dependency and `run.bat` configuration

The styled OSM mask generation step uses QGIS libraries and must be launched through `run.bat`. A separate project environment is not required for this step if `run.bat` correctly points to the local QGIS installation.

The following libraries are loaded from the QGIS installation during the styled rendering step:

```text
qgis.core
PyQt5
osgeo.gdal
```

These packages should not normally be installed separately with `pip`. They are provided by QGIS and initialized by the QGIS batch environment files such as `o4w_env.bat` and `python-qgis.bat` or `python-qgis-ltr.bat`.

Before generating styled raster/OSM tiles, open `run.bat` in a text editor and manually update the required directory paths.

The following paths should be checked and updated according to the local project directory structure:

```bat
set "QGIS_DIR=C:\Program Files\QGIS 3.44.11"
set "MY_SCRIPT=C:\path\to\codes\osm_style_processor.py"
```

Important notes:

- `QGIS_DIR` must point to the main QGIS installation folder.
- `MY_SCRIPT` must point to `osm_style_processor.py`.
- Raster input is not fixed in `run.bat`; it is usually provided from `VHR_OSM_STYLED_DATASET_PREP.ipynb`.

---

## Script roles

| File | Purpose |
|---|---|
| `batch_runner.py` | Batch launcher for OSM downloading. It scans raster/vector inputs and calls `osm_processor.py` for each file. |
| `osm_processor.py` | Downloads OSM layers from the input raster/vector extent using OSMnx and saves them as shapefiles. |
| `batch_formatter.py` | Batch launcher for OSM attribute formatting. It calls `osm_formatter_engine.py` for each downloaded OSM folder. |
| `osm_formatter_engine.py` | Converts raw OSM attributes into a standard schema used by the QML style files. |
| `merge_shp.py` | Merges shapefiles with the same OSM layer name and removes duplicates using `osm_id` or geometry. |
| `osm_style_processor.py` | Uses QGIS in headless mode to render styled OSM masks aligned with raster tiles. |
| `cleanup_tiles.py` | Removes or moves raster/OSM tile pairs whose raster tile contains a black-pixel ratio above the selected threshold. |
| `run.bat` | Loads the QGIS environment and executes `osm_style_processor.py`. |
| `VHR_OSM_STYLED_DATASET_PREP.ipynb` | Main notebook used to organize and execute the dataset preparation workflow. |



## OSM source tags used for layer download

The OSM download step uses predefined OSM tags and geometry filters.

| Output layer | OSM tags used |
|---|---|
| `gis_osm_buildings_a_free_1` | `building=True` |
| `gis_osm_landuse_a_free_1` | `landuse=True` |
| `gis_osm_natural_a_free_1` | `natural=True`, polygon geometries |
| `gis_osm_natural_free_1` | `natural=True`, point/line geometries |
| `gis_osm_places_a_free_1` | `place=True`, polygon geometries |
| `gis_osm_places_free_1` | `place=True`, point geometries |
| `gis_osm_pofw_a_free_1` | `amenity=place_of_worship`, polygon geometries |
| `gis_osm_pofw_free_1` | `amenity=place_of_worship`, point geometries |
| `gis_osm_pois_a_free_1` | `amenity`, `shop`, `tourism`, `leisure`, `office`, `healthcare` |
| `gis_osm_pois_free_1` | `amenity`, `shop`, `tourism`, `leisure`, `office`, `healthcare` |
| `gis_osm_railways_free_1` | `railway=True` |
| `gis_osm_roads_free_1` | `highway=True` |
| `gis_osm_traffic_a_free_1` | `amenity=parking/fuel`, `highway=services/rest_area`, `railway=station/halt` |
| `gis_osm_traffic_free_1` | `amenity=parking/fuel`, `highway=bus_stop/motorway_junction/traffic_signals`, `railway=station/halt/tram_stop` |
| `gis_osm_transport_a_free_1` | `amenity=bus_station/ferry_terminal`, `aeroway=aerodrome/terminal`, `landuse=railway` |
| `gis_osm_transport_free_1` | `amenity=bus_station/ferry_terminal`, `aeroway=aerodrome/terminal/gate`, `public_transport=True` |
| `gis_osm_water_a_free_1` | `natural=water`, `landuse=reservoir`, `waterway=riverbank` |
| `gis_osm_waterways_free_1` | `waterway=True` |
| `gis_osm_political_boundaries_a_free_1` | `boundary=political`; this layer is later merged into the landuse layer during formatting. |

---

## QML style files and styled OSM attributes

The `styles/` directory contains 17 QML files. The QML filenames must match the corresponding shapefile base names. These style files are used by `osm_style_processor.py` to render RGB OSM mask tiles.

| QML file | Styled attribute(s) | Styled values / conditions |
|---|---|---|
| `gis_osm_buildings_a_free_1.qml` | `type` | `cathedral`, `chapel`, `church`, `civic`, `government`, `hospital`, `kindergarten`, `kiosk`, `mosque`, `public`, `religious`, `retail`, `school`, `shop`, `shrine`, `supermarket`, `synagogue`, `temple`, `toilet`, `train_station`, `transportation`, `university` |
| `gis_osm_landuse_a_free_1.qml` | `fclass` | `allotments`, `cemetery`, `commercial`, `forest`, `grass`, `grassland`, `grave_yard`, `heath`, `industrial`, `meadow`, `military`, `nature_reserve`, `orchard`, `park`, `political`, `quarry`, `recreation_ground`, `residential`, `retail`, `scrub`, `village_green`, `vineyard`, `wood` |
| `gis_osm_natural_a_free_1.qml` | `fclass` | `beach`, `cliff`, `grassland`, `scrub` |
| `gis_osm_natural_free_1.qml` | `fclass` | `peak` |
| `gis_osm_places_free_1.qml` | `fclass`, `population` | `city`, `national_capital`, `suburb`, `town`, `village`; population-dependent symbol rules are also present. |
| `gis_osm_pofw_a_free_1.qml` | `fclass` | Place-of-worship polygon styling. |
| `gis_osm_pofw_free_1.qml` | `fclass` | Place-of-worship point styling. |
| `gis_osm_pois_a_free_1.qml` | `fclass` | `alpine_hut`, `arts_centre`, `atm`, `attraction`, `bakery`, `bank`, `bar`, `beverages`, `bicycle_rental`, `bicycle_shop`, `biergarten`, `bookshop`, `butcher`, `cafe`, `camp_site`, `car_dealership`, `car_rental`, `caravan_site`, `castle`, `chalet`, `chemist`, `cinema`, `clothes`, `college`, `community_centre`, `computer_shop`, `convenience`, `courthouse`, `dentist`, `department_store`, `doctors`, `embassy`, `emergency_phone`, `fast_food`, `fire_station`, `florist`, `food_court`, `fort`, `fountain`, `furniture_shop`, `garden`, `garden_centre`, `gift_shop`, `greengrocer`, `guesthouse`, `hairdresser`, `hospital`, `hostel`, `hotel`, `hunting_stand`, `ice_rink`, `jeweller`, `kindergarten`, `kiosk`, `laundry`, `library`, `mall`, `memorial`, `mobile_phone_shop`, `monument`, `motel`, `museum`, `newsagent`, `nightclub`, `observation_tower`, `optician`, `outdoor_shop`, `park`, `pharmacy`, `pitch`, `police`, `post_box`, `post_office`, `prison`, `pub`, `public_building`, `recycling`, `restaurant`, `school`, `shoe_shop`, `sports_centre`, `sports_shop`, `stadium`, `stationery`, `supermarket`, `swimming_pool`, `telephone`, `theatre`, `theme_park`, `toilet`, `tourist_info`, `tower`, `town_hall`, `toy_shop`, `travel_agent`, `university`, `vending_any`, `veterinary`, `video_shop`, `viewpoint`, `wayside_shrine`, `zoo` |
| `gis_osm_pois_free_1.qml` | `fclass` | `arts_centre`, `atm`, `attraction`, `bakery`, `bank`, `bar`, `beauty_shop`, `beverages`, `bicycle_rental`, `bicycle_shop`, `biergarten`, `bookshop`, `cafe`, `camp_site`, `car_rental`, `car_sharing`, `caravan_site`, `castle`, `chemist`, `cinema`, `clothes`, `college`, `community_centre`, `computer_shop`, `convenience`, `dentist`, `department_store`, `doctors`, `dog_park`, `doityourself`, `drinking_water`, `embassy`, `fast_food`, `fire_station`, `florist`, `food_court`, `furniture_shop`, `garden_centre`, `gift_shop`, `golf_course`, `graveyard`, `greengrocer`, `guesthouse`, `hairdresser`, `hospital`, `hostel`, `hotel`, `jeweller`, `library`, `lighthouse`, `mall`, `mobile_phone_shop`, `motel`, `museum`, `nightclub`, `optician`, `outdoor_shop`, `park`, `pharmacy`, `picnic_site`, `pitch`, `playground`, `police`, `post_box`, `post_office`, `pub`, `public_building`, `recycling`, `recycling_clothes`, `recycling_glass`, `recycling_metal`, `recycling_paper`, `restaurant`, `school`, `shelter`, `shoe_shop`, `sports_centre`, `sports_shop`, `supermarket`, `swimming_pool`, `telephone`, `theatre`, `toilet`, `tourist_info`, `town_hall`, `toy_shop`, `university`, `veterinary`, `video_shop`, `viewpoint`, `waste_basket`, `windmill`, `zoo` |
| `gis_osm_railways_free_1.qml` | `fclass` | `light_rail`, `rail`, `subway`, `tram` |
| `gis_osm_roads_free_1.qml` | `fclass`, `bridge`, `tunnel` | Road classes: `bridleway`, `cycleway`, `footway`, `living_street`, `motorway`, `motorway_link`, `path`, `pedestrian`, `primary`, `primary_link`, `residential`, `secondary`, `secondary_link`, `service`, `steps`, `tertiary`, `track`, `track_grade1`, `track_grade2`, `track_grade3`, `track_grade4`, `track_grade5`, `trunk`, `trunk_link`, `unclassified`, `unknown`; bridge/tunnel values: `T`, `F`, `True`, `False`, `yes`, `no` |
| `gis_osm_traffic_a_free_1.qml` | `fclass` | `dam`, `marina`, `weir` |
| `gis_osm_traffic_free_1.qml` | `fclass` | `waterfall` |
| `gis_osm_transport_a_free_1.qml` | `fclass` | `bus_station`, `railway_station` |
| `gis_osm_transport_free_1.qml` | `fclass` | `bus_station`, `bus_stop`, `ferry_terminal`, `railway_halt`, `railway_station`, `taxi`, `tram_stop` |
| `gis_osm_water_a_free_1.qml` | `fclass` | `glacier`, `reservoir`, `river`, `water`, `wetland` |
| `gis_osm_waterways_free_1.qml` | `fclass` | `canal`, `dam`, `derelict_canal`, `ditch`, `drain`, `river`, `stream`, `weir` |

---

## Output structure

For each processed raster, the styled tile generator creates one folder with paired raster and OSM mask tiles:

```text
output_tiles/
└── raster_name/
    ├── raster/
    │   ├── raster_name_patch_0.tif
    │   ├── raster_name_patch_1.tif
    │   └── ...
    └── osm/
        ├── raster_name_mask_0.tif
        ├── raster_name_mask_1.tif
        └── ...
```

The raster and OSM mask tiles have the same tile index, tile size, geotransform, and projection.

---

## Notes

- The `.qml` style files are part of the dataset preparation configuration and should be kept with the project.
- The `styles/` folder must be passed to `run.bat` as `--qml_dir` or defined as `DEFAULT_QML_DIR`.


