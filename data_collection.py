"""Data Collection Script
Step 2: Load DEM, Sentinel-2 bands (NIR/Red), rainfall rasters, and hazard shapefiles.
Organize raw data into data/ subfolders and produce a metadata summary.

Also supports automated download for Ernakulam, Kerala, India:
 - AOI via OSMnx
 - DEM via elevation (SRTM3)
 - Sentinel-2 L2A B04/B08 via Planetary Computer STAC
 - CHIRPS daily rainfall via CHC HTTP
"""
import os
from pathlib import Path
import argparse
from datetime import datetime, timedelta
import gzip
import shutil
import rasterio
from rasterio.mask import mask as rio_mask
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

# Optional libs for automation
try:
    import osmnx as ox
except Exception:  # pragma: no cover
    ox = None
try:
    import planetary_computer as pc
    from pystac_client import Client
except Exception:  # pragma: no cover
    pc = None
    Client = None
import requests
from tqdm import tqdm

DATA_DIR = Path('data')
DEM_DIR = DATA_DIR / 'dem'
SENTINEL_DIR = DATA_DIR / 'sentinel'
RAINFALL_DIR = DATA_DIR / 'rainfall'
HAZARD_DIR = DATA_DIR / 'hazards'
AOI_DIR = DATA_DIR / 'aoi'
META_OUT = DATA_DIR / 'metadata_summary.csv'
DEFAULT_AOI_NAME = 'Ernakulam, Kerala, India'
AOI_PATH = AOI_DIR / 'ernakulam.geojson'


def list_rasters(directory: Path):
    """Return list of raster file paths (GeoTIFF)."""
    return [p for p in directory.glob('**/*') if p.suffix.lower() in ['.tif', '.tiff']]


def list_shapefiles(directory: Path):
    """Return list of shapefile main .shp paths."""
    return [p for p in directory.glob('**/*.shp')]


def summarize_raster(path: Path):
    """Extract key metadata from a raster file."""
    try:
        with rasterio.open(path) as src:
            return {
                'path': str(path),
                'crs': str(src.crs),
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0],
                'bounds': src.bounds,
                'transform': tuple(src.transform)  # store affine as tuple
            }
    except Exception as e:
        return {'path': str(path), 'error': str(e)}


def summarize_vector(path: Path):
    """Extract key metadata from a vector (shapefile)."""
    try:
        gdf = gpd.read_file(path)
        return {
            'path': str(path),
            'crs': str(gdf.crs),
            'feature_count': len(gdf),
            'geom_types': ','.join(sorted(gdf.geom_type.unique()))
        }
    except Exception as e:
        return {'path': str(path), 'error': str(e)}


# ---------- Automation helpers ----------

def ensure_dirs():
    for d in [DEM_DIR, SENTINEL_DIR, RAINFALL_DIR, HAZARD_DIR, AOI_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_or_create_aoi(aoi_name: str, aoi_path: Path = AOI_PATH) -> gpd.GeoDataFrame:
    """Fetch AOI polygon from OSM and persist to GeoJSON. Returns GeoDataFrame in EPSG:4326."""
    ensure_dirs()
    if aoi_path.exists():
        gdf = gpd.read_file(aoi_path)
        return gdf.to_crs(4326)
    if ox is None:
        raise RuntimeError("osmnx not installed; cannot fetch AOI. Install and retry.")
    gdf = ox.geocode_to_gdf(aoi_name)
    gdf = gdf.to_crs(4326)
    gdf.to_file(aoi_path, driver='GeoJSON')
    return gdf


def clip_write(src_path: Path, geom_gdf: gpd.GeoDataFrame, out_path: Path):
    """Clip a raster to AOI geometry and write GeoTIFF."""
    with rasterio.open(src_path) as src:
        geoms = [mapping(geom) for geom in geom_gdf.geometry]
        out_image, out_transform = rio_mask(src, geoms, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, 'w', **out_meta) as dst:
            dst.write(out_image)
    return out_path


def download_dem_via_elevation(aoi_gdf: gpd.GeoDataFrame, out_path: Path = DEM_DIR / 'ernakulam_dem.tif', overwrite=False):
    import elevation
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path
    minx, miny, maxx, maxy = aoi_gdf.total_bounds
    elevation.clip(bounds=(minx, miny, maxx, maxy), output=str(out_path), product='SRTM3')
    # mask to AOI for clean edges
    return clip_write(out_path, aoi_gdf, out_path)


def pick_best_cloud_item(items):
    def get_cloud(it):
        props = it.properties or {}
        return props.get('eo:cloud_cover') or props.get('s2:cloud_cover') or 1000
    items_sorted = sorted(items, key=get_cloud)
    return items_sorted[0] if items_sorted else None


def download_sentinel_pc(aoi_gdf: gpd.GeoDataFrame, start_date: str, end_date: str, max_cloud: float = 20.0):
    if pc is None or Client is None:
        raise RuntimeError('planetary-computer/pystac-client not installed')
    client = Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
    geom = mapping(aoi_gdf.unary_union)
    search = client.search(
        collections=['sentinel-2-l2a'],
        intersects=geom,
        datetime=f"{start_date}/{end_date}",
        query={'eo:cloud_cover': {'lte': max_cloud}}
    )
    items = list(search.get_items())
    best = pick_best_cloud_item(items)
    if not best:
        raise RuntimeError('No Sentinel-2 L2A items found for AOI/date/cloud filters')
    best = pc.sign(best)
    for band, out_name in [('B04', 'B04.tif'), ('B08', 'B08.tif')]:
        asset = best.assets.get(band)
        if asset is None:
            raise RuntimeError(f'Missing asset {band} in selected Sentinel item')
        href = asset.href
        out_path = SENTINEL_DIR / out_name
        with rasterio.open(href) as src:
            geoms = [mapping(g) for g in aoi_gdf.geometry]
            out_image, out_transform = rio_mask(src, geoms, crop=True)
            meta = src.meta.copy()
            meta.update({'height': out_image.shape[1], 'width': out_image.shape[2], 'transform': out_transform})
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with rasterio.open(out_path, 'w', **meta) as dst:
                dst.write(out_image)
    return [SENTINEL_DIR / 'B04.tif', SENTINEL_DIR / 'B08.tif']


def daterange(start: datetime, end: datetime):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def download_chirps_daily(aoi_gdf: gpd.GeoDataFrame, start_date: str, end_date: str, skip_existing=True, max_files=None):
    base = 'https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/{year}/chirps-v2.0.{year}.{month:02d}.{day:02d}.tif.gz'
    start = datetime.fromisoformat(start_date)
    end = datetime.fromisoformat(end_date)
    count = 0
    for d in tqdm(list(daterange(start, end)), desc='CHIRPS days'):
        if max_files and count >= max_files:
            break
        year, month, day = d.year, d.month, d.day
        url = base.format(year=year, month=month, day=day)
        gz_out = RAINFALL_DIR / f'chirps_{year}{month:02d}{day:02d}.tif.gz'
        tif_tmp = RAINFALL_DIR / f'chirps_{year}{month:02d}{day:02d}_tmp.tif'
        tif_clip = RAINFALL_DIR / f'chirps_{year}{month:02d}{day:02d}.tif'
        if skip_existing and tif_clip.exists():
            count += 1
            continue
        RAINFALL_DIR.mkdir(parents=True, exist_ok=True)
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code != 200:
                    continue
                with open(gz_out, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception:
            continue
        try:
            with gzip.open(gz_out, 'rb') as f_in, open(tif_tmp, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            clip_write(tif_tmp, aoi_gdf, tif_clip)
        finally:
            try:
                gz_out.unlink(missing_ok=True)
                tif_tmp.unlink(missing_ok=True)
            except Exception:
                pass
        count += 1
    return count


# ---------- CLI and main ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto-download', action='store_true', help='Enable automated downloads for AOI, DEM, Sentinel, CHIRPS')
    parser.add_argument('--aoi', type=str, default=DEFAULT_AOI_NAME, help='AOI name to geocode via OSMnx')
    parser.add_argument('--s2-start', type=str, default='2015-01-01')
    parser.add_argument('--s2-end', type=str, default='2025-03-31')
    parser.add_argument('--s2-cloud', type=float, default=20.0)
    parser.add_argument('--chirps-start', type=str, default='2015-01-01')
    parser.add_argument('--chirps-end', type=str, default='2025-03-31')
    parser.add_argument('--chirps-max', type=int, default=None, help='Limit number of CHIRPS days to fetch (None for all)')
    parser.add_argument('--overwrite-dem', action='store_true')
    args = parser.parse_args()

    # Ensure directory structure exists
    ensure_dirs()

    if args.auto_download:
        # AOI
        aoi_gdf = get_or_create_aoi(args.aoi)
        # DEM
        dem_path = download_dem_via_elevation(aoi_gdf, overwrite=args.overwrite_dem)
        print(f"DEM ready: {dem_path}")
        # Sentinel via Planetary Computer
        try:
            s2_paths = download_sentinel_pc(aoi_gdf, args.s2_start, args.s2_end, args.s2_cloud)
            print(f"Sentinel bands saved: {s2_paths}")
        except Exception as e:
            print(f"Sentinel download failed: {e}")
        # CHIRPS daily
        try:
            n = download_chirps_daily(aoi_gdf, args.chirps_start, args.chirps_end, skip_existing=True, max_files=args.chirps_max)
            print(f"CHIRPS days processed: {n}")
        except Exception as e:
            print(f"CHIRPS download failed: {e}")

    # Collect raster and vector files
    dem_files = list_rasters(DEM_DIR)
    sentinel_files = list_rasters(SENTINEL_DIR)
    rainfall_files = list_rasters(RAINFALL_DIR)
    hazard_vectors = list_shapefiles(HAZARD_DIR)

    records = []

    for f in dem_files:
        rec = summarize_raster(f)
        rec['category'] = 'DEM'
        records.append(rec)
    for f in sentinel_files:
        rec = summarize_raster(f)
        rec['category'] = 'Sentinel'
        records.append(rec)
    for f in rainfall_files:
        rec = summarize_raster(f)
        rec['category'] = 'Rainfall'
        records.append(rec)
    for f in hazard_vectors:
        rec = summarize_vector(f)
        rec['category'] = 'HazardVector'
        records.append(rec)

    if records:
        df = pd.DataFrame(records)
        df.to_csv(META_OUT, index=False)
        print(f"Metadata summary saved to {META_OUT}")
    else:
        print("No data files found yet. Place raw files into the data/ subdirectories.")

    print("Summary counts:")
    print(f" DEM rasters: {len(dem_files)}")
    print(f" Sentinel rasters: {len(sentinel_files)}")
    print(f" Rainfall rasters: {len(rainfall_files)}")
    print(f" Hazard shapefiles: {len(hazard_vectors)}")


if __name__ == '__main__':
    main()
