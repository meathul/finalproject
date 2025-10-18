"""
Extract hazard layers from a PDF map and integrate as raster/vector hazards.

Features:
- Detects GeoPDF pages via GDAL and exports to GeoTIFF
- Clips to AOI (data/aoi/ernakulam.geojson)
- If RGB map, optionally clusters into hazard classes (KMeans)
- Saves outputs into data/hazards/

Usage examples:
  # Default PDF in project root, auto-selects first georeferenced page
  python hazard_from_pdf.py --pdf Ernakulam_Hist.pdf --classify 4

  # Explicit page index (0-based among detected subdatasets)
  python hazard_from_pdf.py --pdf Ernakulam_Hist.pdf --page 0 --classify 3

Outputs (in data/hazards/):
  - hazard_pdf_raw.tif     : exported page
  - hazard_pdf_clip.tif    : clipped to AOI
  - hazard_pdf_class.tif   : single-band clustered classes (optional)
  - hazard_pdf_class.shp   : vector polygons of classes (optional)
"""
from pathlib import Path
import argparse
import sys
import json
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping
import tempfile

# Try PyMuPDF as a fallback renderer
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

# GDAL bindings (installed via conda recommended)
from osgeo import gdal

import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.features import shapes
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from sklearn.cluster import KMeans

DATA_DIR = Path('data')
HAZ_DIR = DATA_DIR / 'hazards'
AOI_DIR = DATA_DIR / 'aoi'
AOI_PATH = AOI_DIR / 'ernakulam.geojson'

RAW_TIF = HAZ_DIR / 'hazard_pdf_raw.tif'
CLIP_TIF = HAZ_DIR / 'hazard_pdf_clip.tif'
CLASS_TIF = HAZ_DIR / 'hazard_pdf_class.tif'
CLASS_SHP = HAZ_DIR / 'hazard_pdf_class.shp'


def ensure_dirs():
    HAZ_DIR.mkdir(parents=True, exist_ok=True)
    AOI_DIR.mkdir(parents=True, exist_ok=True)


def load_aoi(aoi_path: Path) -> gpd.GeoDataFrame:
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI not found at {aoi_path}. Run data_collection.py --auto-download once.")
    gdf = gpd.read_file(aoi_path)
    return gdf.to_crs(4326)


def find_pdf_subdatasets(pdf_path: Path):
    ds = gdal.OpenEx(str(pdf_path), gdal.OF_READONLY | gdal.OF_RASTER)
    if ds is None:
        raise RuntimeError(f"GDAL cannot open PDF (no PDF driver or file not supported): {pdf_path}")
    subs = ds.GetSubDatasets() or []
    return subs


def export_pdf_page_to_tif(pdf_path: Path, out_path: Path, page_index: int | None = None) -> Path:
    """Use GDAL to export a GeoPDF page (or first georeferenced page) to GeoTIFF."""
    subs = find_pdf_subdatasets(pdf_path)
    src_name = None
    if subs:
        # Prefer a subdataset with georeference (description often includes 'Page')
        candidates = [s[0] for s in subs]
        if page_index is not None and 0 <= page_index < len(candidates):
            src_name = candidates[page_index]
        else:
            src_name = candidates[0]
    else:
        # No subdatasets; try direct
        src_name = f"PDF:{pdf_path}"
    # Translate to GeoTIFF
    gdal.Translate(destName=str(out_path), srcDS=src_name, format='GTiff')
    if not out_path.exists():
        raise RuntimeError('Failed to export PDF page to GeoTIFF (not georeferenced or driver missing)')
    return out_path


def render_pdf_with_pymupdf(pdf_path: Path, out_tif: Path, page_index: int | None, dpi: int, extent: tuple[float, float, float, float], epsg: int):
    """Render a PDF page to a raster and assign georeferencing using provided extent and EPSG."""
    if fitz is None:
        raise RuntimeError('PyMuPDF not installed. Install with: pip install pymupdf')
    doc = fitz.open(str(pdf_path))
    pg = page_index if page_index is not None else 0
    if pg < 0 or pg >= len(doc):
        raise IndexError(f"Page index out of range. PDF has {len(doc)} pages")
    page = doc.load_page(pg)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    width, height = pix.width, pix.height
    # Save to a temporary PNG then write GeoTIFF with transform
    with tempfile.TemporaryDirectory() as td:
        png_path = Path(td) / 'page.png'
        pix.save(str(png_path))
        with rasterio.open(png_path) as tmp:
            img = tmp.read()
            profile = tmp.profile
        transform = from_bounds(*extent, width=width, height=height)
        profile.update(driver='GTiff', transform=transform, crs=f'EPSG:{epsg}', dtype=img.dtype, count=img.shape[0])
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, 'w', **profile) as dst:
            dst.write(img)
    return out_tif


def clip_to_aoi(in_tif: Path, aoi_gdf: gpd.GeoDataFrame, out_tif: Path) -> Path:
    # Clip using rasterio, reproject AOI to raster CRS
    with rasterio.open(in_tif) as src:
        aoi = aoi_gdf
        if aoi.crs and src.crs and aoi.crs != src.crs:
            aoi = aoi.to_crs(src.crs)
        geoms = [mapping(geom.buffer(0)) for geom in aoi.geometry]
        out, transform = rio_mask(src, geoms, crop=True)
        meta = src.meta.copy()
        meta.update({
            'height': out.shape[1],
            'width': out.shape[2],
            'transform': transform
        })
        out_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_tif, 'w', **meta) as dst:
            dst.write(out)
    return out_tif


def kmeans_classify_rgb(in_tif: Path, out_tif: Path, n_classes: int = 4, sample_frac: float = 0.1) -> Path:
    with rasterio.open(in_tif) as src:
        if src.count < 3:
            raise ValueError('Expected RGB raster (3 bands) for clustering')
        # Read RGB (first 3 bands)
        arr = src.read([1, 2, 3]).astype(np.float32)
        h, w = arr.shape[1:]
        rgb = arr.reshape(3, -1).T  # (N,3)
        # Sample for speed
        N = rgb.shape[0]
        idx = np.random.choice(N, size=max(5000, int(N * sample_frac)), replace=False)
        rgb_sample = rgb[idx]
        kmeans = KMeans(n_clusters=n_classes, random_state=42)
        kmeans.fit(rgb_sample)
        labels = kmeans.predict(rgb)
        labels_img = labels.reshape(h, w).astype(np.uint8)
        meta = src.meta.copy()
        meta.update({'count': 1, 'dtype': 'uint8'})
        with rasterio.open(out_tif, 'w', **meta) as dst:
            dst.write(labels_img, 1)
    return out_tif


def vectorize_classes(in_tif: Path, out_shp: Path):
    with rasterio.open(in_tif) as src:
        image = src.read(1)
        mask = np.ones_like(image, dtype=bool)
        results = (
            (geom, int(val))
            for geom, val in shapes(image, mask=mask, transform=src.transform)
        )
        geoms, vals = zip(*results) if image.size else ([], [])
        if not geoms:
            print('No polygons produced from classes.')
            return
        gdf = gpd.GeoDataFrame({'hazard_cls': list(vals)}, geometry=[gpd.GeoSeries.from_wkt(json.dumps(g)) for g in geoms], crs=src.crs)
        # The above line using from_wkt is not correct for GeoJSON; construct via shapely


def vectorize_classes_shapely(in_tif: Path, out_shp: Path):
    with rasterio.open(in_tif) as src:
        image = src.read(1)
        mask = np.ones_like(image, dtype=bool)
        feats = list(shapes(image, mask=mask, transform=src.transform))
        if not feats:
            print('No polygons produced from classes.')
            return
        geoms = [gpd.GeoSeries.from_wkt(None)]


def vectorize_classes_safe(in_tif: Path, out_shp: Path):
    # Proper vectorization using shapely geometry from rasterio.features.shapes
    from shapely.geometry import shape
    with rasterio.open(in_tif) as src:
        image = src.read(1)
        mask = np.ones_like(image, dtype=bool)
        feat_iter = shapes(image, mask=mask, transform=src.transform)
        geoms = []
        vals = []
        for geom, val in feat_iter:
            geoms.append(shape(geom))
            vals.append(int(val))
        if not geoms:
            print('No polygons produced from classes.')
            return
        gdf = gpd.GeoDataFrame({'hazard_cls': vals}, geometry=geoms, crs=src.crs)
        out_shp.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out_shp)
        print(f'Saved vectorized classes to {out_shp}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str, default='Ernakulam_Hist.pdf', help='Path to hazard PDF')
    parser.add_argument('--page', type=int, default=None, help='Subdataset/page index to export (auto if omitted)')
    parser.add_argument('--classify', type=int, default=0, help='If >0, cluster RGB into this many classes')
    parser.add_argument('--extent', type=str, default=None, help='If PDF is not georeferenced, provide extent as "minx,miny,maxx,maxy" (EPSG given by --epsg)')
    parser.add_argument('--epsg', type=int, default=4326, help='EPSG for provided extent (default 4326)')
    parser.add_argument('--dpi', type=int, default=300, help='Render DPI for non-georef PDFs (PyMuPDF fallback)')
    args = parser.parse_args()

    ensure_dirs()
    aoi = load_aoi(AOI_PATH)

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f'PDF not found: {pdf_path}')
        sys.exit(1)

    # Try GeoPDF export first; fallback to PyMuPDF render + manual extent
    did_export = False
    try:
        export_pdf_page_to_tif(pdf_path, RAW_TIF, page_index=args.page)
        did_export = True
        print(f'Exported PDF page to {RAW_TIF}')
    except Exception as e:
        print(f'GDAL PDF export failed: {e}')
        if not args.extent:
            print('Provide --extent "minx,miny,maxx,maxy" and optionally --page/--dpi to render and georeference the PDF.')
            sys.exit(1)
        try:
            minx, miny, maxx, maxy = map(float, args.extent.split(','))
        except Exception:
            print('Invalid --extent format. Use: --extent minx,miny,maxx,maxy')
            sys.exit(1)
        render_pdf_with_pymupdf(pdf_path, RAW_TIF, args.page, args.dpi, (minx, miny, maxx, maxy), args.epsg)
        did_export = True
        print(f'Rendered and georeferenced PDF page to {RAW_TIF}')

    # Clip to AOI
    if did_export:
        clip_to_aoi(RAW_TIF, aoi, CLIP_TIF)
        print(f'Clipped hazard raster to AOI: {CLIP_TIF}')

    # Optional: classify RGB into hazard classes
    if args.classify and args.classify > 0 and did_export:
        try:
            kmeans_classify_rgb(CLIP_TIF, CLASS_TIF, n_classes=args.classify)
            print(f'Wrote hazard classes raster: {CLASS_TIF}')
            vectorize_classes_safe(CLASS_TIF, CLASS_SHP)
        except Exception as e:
            print(f'Clustering/vectorizing failed: {e}')

if __name__ == '__main__':
    main()
