import elevation
import argparse
from pathlib import Path
import sys

# Kerala bounding box (approx): (74.5, 8.0, 77.5, 12.9)
DEFAULT_BOUNDS = (74.5, 8.0, 77.5, 12.9)

# Output path
default_out = Path('data/dem/kerala_dem.tif')


def ensure_dirs(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def download_dem(bounds, out_path: Path, product='SRTM3', overwrite=False):
    ensure_dirs(out_path)
    if out_path.exists() and not overwrite:
        print(f"DEM already exists at {out_path}. Use --overwrite to refresh.")
        return out_path
    print(f"Downloading DEM product={product} bounds={bounds} -> {out_path}")
    elevation.clip(bounds=bounds, output=str(out_path), product=product)
    print("Download complete.")
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Download and clip DEM (SRTM via elevation).')
    parser.add_argument('--minlon', type=float, default=DEFAULT_BOUNDS[0])
    parser.add_argument('--minlat', type=float, default=DEFAULT_BOUNDS[1])
    parser.add_argument('--maxlon', type=float, default=DEFAULT_BOUNDS[2])
    parser.add_argument('--maxlat', type=float, default=DEFAULT_BOUNDS[3])
    parser.add_argument('--product', type=str, default='SRTM3', choices=['SRTM3', 'SRTM1'])
    parser.add_argument('--out', type=str, default=str(default_out))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    bounds = (args.minlon, args.minlat, args.maxlon, args.maxlat)
    out_path = Path(args.out)

    try:
        download_dem(bounds, out_path, product=args.product, overwrite=args.overwrite)
    except Exception as e:
        print(f"Failed to download DEM: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
