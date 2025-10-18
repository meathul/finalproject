"""Preprocessing & Feature Engineering
Step 3 tasks:
 - Derive slope and terrain metrics from DEM (using richdem)
 - Compute NDVI from Sentinel-2 (Band 8 NIR, Band 4 Red)
 - Resample rainfall raster(s) to DEM resolution
 - Rasterize hazard shapefile(s) to DEM grid
 - Merge all aligned layers into a single DataFrame and export CSV

Assumptions:
 - There is at least one DEM raster in data/dem
 - Sentinel band filenames contain indicators like B04 (red) and B08 (nir)
 - Rainfall: single or multiple rasters (if multiple, mean will be taken)
 - Hazard shapefile contains a categorical risk or binary hazard field (e.g., 'hazard' or 'risk')

Outputs:
 - data/processed/aligned_features.csv with columns:
   x, y, elevation, slope, ndvi, rainfall, hazard_label
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject
import geopandas as gpd

DATA_DIR = Path('data')
DEM_DIR = DATA_DIR / 'dem'
SENTINEL_DIR = DATA_DIR / 'sentinel'
RAINFALL_DIR = DATA_DIR / 'rainfall'
HAZARD_DIR = DATA_DIR / 'hazards'
PROCESSED_DIR = DATA_DIR / 'processed'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = PROCESSED_DIR / 'aligned_features.csv'

# CONFIG: hazard attribute field name candidates
HAZARD_ATTR_CANDIDATES = ['hazard', 'risk', 'class', 'label']


def pick_first(path_list):
    return path_list[0] if path_list else None


def find_dem():
    files = list(DEM_DIR.glob('*.tif'))
    return pick_first(files)


def find_sentinel_band(patterns):
    for p in SENTINEL_DIR.glob('*.tif'):
        name = p.name.lower()
        if all(pt.lower() in name for pt in patterns):
            return p
    return None


def load_raster(path):
    return rasterio.open(path)


def compute_slope(dem_path):
    """Compute slope in degrees using numpy gradient (Horn-style approximation)."""
    with rasterio.open(dem_path) as src:
        dem = src.read(1, masked=True).filled(np.nan).astype(np.float32)
        profile = src.profile
        transform = src.transform
    # Pixel size
    xres = transform.a
    yres = abs(transform.e)
    # Gradients (dz/dy first axis, dz/dx second axis)
    dz_dy, dz_dx = np.gradient(dem, yres, xres)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg, profile


def compute_ndvi(nir_path, red_path, target_profile):
    with rasterio.open(nir_path) as nir_src, rasterio.open(red_path) as red_src:
        # Reproject/resample if needed to target grid (DEM)
        nir = np.empty((target_profile['height'], target_profile['width']), dtype=np.float32)
        red = np.empty_like(nir)
        reproject(
            source=rasterio.band(nir_src, 1),
            destination=nir,
            src_transform=nir_src.transform,
            src_crs=nir_src.crs,
            dst_transform=target_profile['transform'],
            dst_crs=target_profile['crs'],
            dst_height=target_profile['height'],
            dst_width=target_profile['width'],
            resampling=Resampling.bilinear
        )
        reproject(
            source=rasterio.band(red_src, 1),
            destination=red,
            src_transform=red_src.transform,
            src_crs=red_src.crs,
            dst_transform=target_profile['transform'],
            dst_crs=target_profile['crs'],
            dst_height=target_profile['height'],
            dst_width=target_profile['width'],
            resampling=Resampling.bilinear
        )
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi[np.isinf(ndvi)] = np.nan
    return ndvi


def aggregate_rainfall(target_profile):
    rainfall_files = list(RAINFALL_DIR.glob('*.tif'))
    if not rainfall_files:
        return np.full((target_profile['height'], target_profile['width']), np.nan, dtype=np.float32)
    accum = np.zeros((target_profile['height'], target_profile['width']), dtype=np.float32)
    count = 0
    for rf in rainfall_files:
        with rasterio.open(rf) as rsrc:
            arr = np.empty((target_profile['height'], target_profile['width']), dtype=np.float32)
            reproject(
                source=rasterio.band(rsrc, 1),
                destination=arr,
                src_transform=rsrc.transform,
                src_crs=rsrc.crs,
                dst_transform=target_profile['transform'],
                dst_crs=target_profile['crs'],
                dst_height=target_profile['height'],
                dst_width=target_profile['width'],
                resampling=Resampling.bilinear
            )
            mask = ~np.isnan(arr)
            accum[mask] += arr[mask]
            count += 1
    if count > 0:
        accum /= count
    accum[accum == 0] = np.nan  # if zeros are nodata
    return accum


def rasterize_hazards(target_profile):
    shapefiles = list(HAZARD_DIR.glob('*.shp'))
    if not shapefiles:
        return np.full((target_profile['height'], target_profile['width']), np.nan, dtype=np.float32)

    # Merge all hazard layers into one GeoDataFrame
    gdfs = []
    for shp in shapefiles:
        try:
            gdf = gpd.read_file(shp)
            gdfs.append(gdf)
        except Exception as e:
            print(f"Failed to read {shp}: {e}")
    if not gdfs:
        return np.full((target_profile['height'], target_profile['width']), np.nan, dtype=np.float32)

    hazards = pd.concat(gdfs, ignore_index=True)

    # Identify hazard attribute
    hazard_field = None
    cols_lower = {c.lower(): c for c in hazards.columns}
    for cand in HAZARD_ATTR_CANDIDATES:
        if cand in cols_lower:
            hazard_field = cols_lower[cand]
            break
    if hazard_field is None:
        # create a default binary field
        hazards['hazard_tmp'] = 1
        hazard_field = 'hazard_tmp'

    # Reproject to DEM CRS if needed
    dem_crs = target_profile['crs']
    if hazards.crs and hazards.crs != dem_crs:
        hazards = hazards.to_crs(dem_crs)

    shapes = ((geom, val) for geom, val in zip(hazards.geometry, hazards[hazard_field]))

    out = rasterize(
        shapes=shapes,
        out_shape=(target_profile['height'], target_profile['width']),
        transform=target_profile['transform'],
        fill=np.nan,
        dtype='float32'
    )
    return out


def build_dataframe(elevation, slope, ndvi, rainfall, hazard, profile):
    # Create coordinate arrays (center of pixels)
    transform = profile['transform']
    height, width = elevation.shape
    rows, cols = np.indices((height, width))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.array(xs).ravel()
    ys = np.array(ys).ravel()

    df = pd.DataFrame({
        'x': xs,
        'y': ys,
        'elevation': elevation.ravel(),
        'slope': slope.ravel(),
        'ndvi': ndvi.ravel(),
        'rainfall': rainfall.ravel(),
        'hazard_label': hazard.ravel()
    })

    # Drop rows with all NaN features or missing hazard label
    df = df.dropna(subset=['elevation', 'slope', 'ndvi', 'rainfall', 'hazard_label'], how='any')
    return df


def main():
    dem_path = find_dem()
    if not dem_path:
        raise FileNotFoundError("No DEM found in data/dem/")

    # Compute slope & load elevation
    slope, dem_profile = compute_slope(dem_path)
    with rasterio.open(dem_path) as dem_src:
        elevation = dem_src.read(1).astype(np.float32)

    # Find sentinel bands
    nir_band = find_sentinel_band(['b08'])
    red_band = find_sentinel_band(['b04'])
    if not nir_band or not red_band:
        print("Warning: Missing Sentinel-2 NIR or Red band. NDVI will be NaN.")
        ndvi = np.full_like(elevation, np.nan, dtype=np.float32)
    else:
        ndvi = compute_ndvi(nir_band, red_band, dem_profile)

    # Rainfall aggregate
    rainfall = aggregate_rainfall(dem_profile)

    # Hazard rasterization
    hazard = rasterize_hazards(dem_profile)

    # Build DataFrame
    df = build_dataframe(elevation, slope, ndvi, rainfall, hazard, dem_profile)
    if df.empty:
        print("Resulting DataFrame is empty. Check data sources.")
    else:
        df.to_csv(OUT_CSV, index=False)
        print(f"Saved feature dataset to {OUT_CSV} with {len(df)} rows.")


if __name__ == '__main__':
    main()
