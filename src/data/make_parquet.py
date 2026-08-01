from pathlib import Path
import polars as pl

def make_parquet():
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[1] 
    
    raw_data_path = str(project_root / "data" / "raw" / "characteristics_by_month" / "*.csv")
    output_path = project_root / "data" / "processed" / "chars.parquet"
    
    (
        pl.scan_csv(raw_data_path, infer_schema_length=10000)
        .sort(["DATE", "permno"])
        .sink_parquet(output_path)
    )

if __name__ == "__main__":
    make_parquet()