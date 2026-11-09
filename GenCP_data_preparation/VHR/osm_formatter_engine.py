"""
osm_formatter_engine.py – OSM Shapefile Formatting Engine

This script formats downloaded OSM shapefiles in a single input folder.
It reads each supported OSM layer, converts its attributes into a predefined
standard schema, assigns class names and numeric codes where applicable, and
writes the formatted shapefiles to the output folder.

Input parameters:
--input / -i:
Folder containing the raw OSM shapefiles.

--output / -o:
    Folder where the formatted shapefiles will be saved.

Outputs:
- Formatted shapefiles with standardized attribute columns.

"""


import os
import pandas as pd
import geopandas as gpd
import warnings
import argparse

# For log cleanup
pd.set_option('future.no_silent_downcasting', True)


# ---------------------------------------------------------------------------
# 1. TARGET SCHEMAS
# ---------------------------------------------------------------------------

STD_COLS = ["osm_id", "code", "fclass", "name"]

LAYER_SCHEMAS = {
    "gis_osm_buildings_a_free_1": STD_COLS + ["type"],
    "gis_osm_roads_free_1": STD_COLS + ["ref", "oneway", "maxspeed", "layer", "bridge", "tunnel"],
    "gis_osm_railways_free_1": STD_COLS + ["layer", "bridge", "tunnel"],
    "gis_osm_places_a_free_1": STD_COLS + ["population"],
    "gis_osm_places_free_1":   STD_COLS + ["population"],
    "gis_osm_waterways_free_1": STD_COLS + ["width"],
    "gis_osm_landuse_a_free_1": STD_COLS,
    "gis_osm_natural_a_free_1": STD_COLS,
    "gis_osm_natural_free_1":   STD_COLS,
    "gis_osm_pofw_a_free_1":    STD_COLS,
    "gis_osm_pofw_free_1":      STD_COLS,
    "gis_osm_pois_a_free_1":    STD_COLS,
    "gis_osm_pois_free_1":      STD_COLS,
    "gis_osm_traffic_a_free_1": STD_COLS,
    "gis_osm_traffic_free_1":   STD_COLS,
    "gis_osm_transport_a_free_1": STD_COLS,
    "gis_osm_transport_free_1":   STD_COLS,
    "gis_osm_water_a_free_1":     STD_COLS,
}

# ---------------------------------------------------------------------------
# 2. UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def init_gdf(source_gdf, schema_cols):
    out = gpd.GeoDataFrame(columns=schema_cols + ["geometry"], crs=source_gdf.crs)
    out["geometry"] = source_gdf["geometry"]
    
    if "osmid" in source_gdf.columns:
        out["osm_id"] = source_gdf["osmid"].astype(str)
    elif "osm_id" in source_gdf.columns:
        out["osm_id"] = source_gdf["osm_id"].astype(str)
    else:
        out["osm_id"] = source_gdf.index.astype(str)
        
    out["code"] = 0
    out["fclass"] = ""
    out["name"] = ""
    return out

def safe_get(gdf, col_name, default_val=""):
    if col_name in gdf.columns:
        return gdf[col_name].fillna(default_val)
    return pd.Series([default_val] * len(gdf), index=gdf.index)

# ---------------------------------------------------------------------------
# 3. LAYER PROCESSING RULES
# ---------------------------------------------------------------------------

def process_buildings(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["fclass"] = "building"
    out["code"] = 1500
    out["name"] = safe_get(b_gdf, "name").astype(str)
    out["type"] = safe_get(b_gdf, "building").astype(str)
    return out

def process_roads(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    out["ref"] = safe_get(b_gdf, "ref").astype(str)
    
    if "highway" in b_gdf.columns:
        out["fclass"] = b_gdf["highway"].fillna("").astype(str)
    
    out["oneway"] = safe_get(b_gdf, "oneway", "B").astype(str)

    maxspeed_raw = safe_get(b_gdf, "maxspeed", 0).astype(str)
    maxspeed_num = maxspeed_raw.str.extract(r"(\d+)")[0]
    out["maxspeed"] = pd.to_numeric(maxspeed_num, errors="coerce").fillna(0).astype(int)

    out["layer"] = pd.to_numeric(
        safe_get(b_gdf, "layer", 0),
        errors="coerce"
    ).fillna(0).astype(int)

    out["bridge"] = safe_get(b_gdf, "bridge", "F").astype(str)
    out["tunnel"] = safe_get(b_gdf, "tunnel", "F").astype(str)
    return out



def process_railways(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    out["layer"] = pd.to_numeric(safe_get(b_gdf, "layer", 0), errors='coerce').fillna(0).astype(int)
    out["bridge"] = safe_get(b_gdf, "bridge", "F").astype(str)
    out["tunnel"] = safe_get(b_gdf, "tunnel", "F").astype(str)
    
    if "railway" in b_gdf.columns:
        out["fclass"] = b_gdf["railway"].fillna("").astype(str)
        
    mask_rail = out["fclass"] == "rail"
    out.loc[mask_rail, "code"] = 6101
    mask_sub = out["fclass"] == "subway"
    out.loc[mask_sub, "code"] = 6103
    mask_tram = out["fclass"] == "tram"
    out.loc[mask_tram, "code"] = 6104
    return out

def process_water(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    
    if "natural" in b_gdf.columns:
        mask = b_gdf["natural"] == "water"
        out.loc[mask, "fclass"] = "water"
        out.loc[mask, "code"] = 8200
        mask_other = (b_gdf["natural"].notna()) & (b_gdf["natural"] != "water")
        out.loc[mask_other, "fclass"] = b_gdf.loc[mask_other, "natural"]

    if "waterway" in b_gdf.columns:
        mask = b_gdf["waterway"] == "riverbank"
        out.loc[mask, "fclass"] = "riverbank"
        out.loc[mask, "code"] = 8202

    if "landuse" in b_gdf.columns:
        mask = b_gdf["landuse"] == "reservoir"
        out.loc[mask, "fclass"] = "reservoir"
        out.loc[mask, "code"] = 8201
    return out

def process_waterways_lines(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    if "waterway" in b_gdf.columns:
        out["fclass"] = b_gdf["waterway"].fillna("").astype(str)
    # FIX: Use to_numeric to first convert decimal values (e.g., 1.5) to float, then convert them to int.
    out["width"] = pd.to_numeric(safe_get(b_gdf, "width", 0), errors='coerce').fillna(0).astype(int)
    return out

def process_places(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    if "place" in b_gdf.columns:
        out["fclass"] = b_gdf["place"].fillna("").astype(str)
    # FIX: Protect against possible decimal or invalid formats in population data.
    out["population"] = pd.to_numeric(safe_get(b_gdf, "population", 0), errors='coerce').fillna(0).astype(int)
    return out

def process_generic_landuse(b_gdf, target_cols, input_folder=None):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)

    if "landuse" in b_gdf.columns:
        out["fclass"] = b_gdf["landuse"].fillna("").astype(str)

    if input_folder:
        poly_path = os.path.join(input_folder, "gis_osm_political_boundaries_a_free_1.shp")

        if os.path.exists(poly_path):
            print("    -> Boundary data are being added to the landuse layer.")

            poly_gdf = gpd.read_file(poly_path)

            if not poly_gdf.empty:
                poly_out = init_gdf(poly_gdf, target_cols)
                poly_out["name"] = safe_get(poly_gdf, "name").astype(str)

                # Boundary polygons are rendered through the landuse style.
                # Political and administrative boundaries are mapped to the
                # existing political class to avoid unstyled/black QML output.
                poly_out["fclass"] = "political"

                out = pd.concat([out, poly_out], ignore_index=True)
                out = gpd.GeoDataFrame(out, crs=b_gdf.crs)

    return out

def process_generic_natural(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    if "natural" in b_gdf.columns:
        out["fclass"] = b_gdf["natural"].fillna("").astype(str)
    return out

def process_generic_pois(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)

    out["fclass"] = ""
    for col in ["leisure", "amenity", "shop", "tourism"]:
        if col in b_gdf.columns:
            vals = b_gdf[col].fillna("").astype(str)
            mask = (out["fclass"] == "") & (vals != "")
            out.loc[mask, "fclass"] = vals[mask]

    return out

def process_generic_pofw(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    out["fclass"] = "place_of_worship"
    if "building" in b_gdf.columns:
        mask = b_gdf["building"].notna()
        out.loc[mask, "fclass"] = "building"
    return out

def process_transport(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    if "amenity" in b_gdf.columns:
        mask = b_gdf["amenity"].isin(["bus_station", "ferry_terminal"])
        out.loc[mask, "fclass"] = b_gdf.loc[mask, "amenity"]
    if "aeroway" in b_gdf.columns:
        mask = b_gdf["aeroway"].notna()
        out.loc[mask, "fclass"] = b_gdf.loc[mask, "aeroway"]
    return out

def process_traffic(b_gdf, target_cols):
    out = init_gdf(b_gdf, target_cols)
    out["name"] = safe_get(b_gdf, "name").astype(str)
    if "amenity" in b_gdf.columns:
        mask = b_gdf["amenity"].isin(["parking", "fuel"])
        out.loc[mask, "fclass"] = b_gdf.loc[mask, "amenity"]
    return out

# ---------------------------------------------------------------------------
# 4. DISPATCHER
# ---------------------------------------------------------------------------

DISPATCHER = {
    "gis_osm_buildings_a_free_1": process_buildings,
    "gis_osm_roads_free_1":       process_roads,
    "gis_osm_railways_free_1":    process_railways,
    "gis_osm_water_a_free_1":     process_water,
    "gis_osm_waterways_free_1":   process_waterways_lines,
    "gis_osm_places_a_free_1":    process_places,
    "gis_osm_places_free_1":      process_places,
    "gis_osm_landuse_a_free_1":   process_generic_landuse,
    "gis_osm_natural_a_free_1":   process_generic_natural,
    "gis_osm_natural_free_1":     process_generic_natural,
    "gis_osm_pois_a_free_1":      process_generic_pois,
    "gis_osm_pois_free_1":        process_generic_pois,
    "gis_osm_pofw_a_free_1":      process_generic_pofw,
    "gis_osm_pofw_free_1":        process_generic_pofw,
    "gis_osm_transport_a_free_1": process_transport,
    "gis_osm_transport_free_1":   process_transport,
    "gis_osm_traffic_a_free_1":   process_traffic,
    "gis_osm_traffic_free_1":     process_traffic
}

def process_single_layer(input_folder, output_folder, filename):

    stem = os.path.splitext(filename)[0]
    
    # Skip the political boundaries layer as a standalone output because it is merged into the landuse layer when available
    if "political_boundaries" in stem:
        return f"[SKIP] Political boundaries are merged into landuse: {filename}"
        
    if stem not in LAYER_SCHEMAS:
        return f"[SKIP] Undefined layer: {filename}"

    target_cols = LAYER_SCHEMAS[stem]
    processor_func = DISPATCHER.get(stem)
    
    if not processor_func:
        return f"[WARN] No processing function found: {filename}"
        
    try:
        in_path = os.path.join(input_folder, filename)
        b_gdf = gpd.read_file(in_path)
        
       
        if stem == "gis_osm_landuse_a_free_1":

            out_gdf = processor_func(b_gdf, target_cols, input_folder=input_folder)
        else:
            if b_gdf.empty:
                out_gdf = init_gdf(b_gdf, target_cols)
            else:
                out_gdf = processor_func(b_gdf, target_cols)
        
        os.makedirs(output_folder, exist_ok=True)
        out_path = os.path.join(output_folder, filename)
        out_gdf.to_file(out_path, encoding='utf-8')
        return f"[OK] {stem} processed (n={len(out_gdf)})"
        
    except Exception as e:
        return f"[ERROR] {stem} processing error: {e}"



def main():
    parser = argparse.ArgumentParser(
        description="Performs OSM shapefile format conversion for a single folder."
    )
    parser.add_argument("--input",  "-i", required=True, help="Folder containing the raw shapefiles")
    parser.add_argument("--output", "-o", required=True, help="Folder where the processed outputs will be saved")
    args = parser.parse_args()

    INPUT_FOLDER = args.input
    OUT_FOLDER   = args.output

    os.makedirs(OUT_FOLDER, exist_ok=True)
    print(f"Input File: {INPUT_FOLDER}")
    print(f"Output File: {OUT_FOLDER}")
    print("-" * 50)

    for fname in sorted(os.listdir(INPUT_FOLDER)):
        if not fname.lower().endswith(".shp"):
            continue
        status = process_single_layer(INPUT_FOLDER, OUT_FOLDER, fname)
        print(status)


if __name__ == "__main__":
    main()