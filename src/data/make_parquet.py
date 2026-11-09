import polars as pl

from src.config import CHARS_PARQUET, PROCESSED_DIR, RAW_CHARS_GLOB

def make_parquet():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_csv(str(RAW_CHARS_GLOB), infer_schema_length=10000)
        .sort(["DATE", "permno"])
        .sink_parquet(CHARS_PARQUET)
    )

if __name__ == "__main__":
    make_parquet()