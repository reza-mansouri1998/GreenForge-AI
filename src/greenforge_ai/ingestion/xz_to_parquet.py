import polars as pl
import lzma
from pathlib import Path
from greenforge_ai.utils.logger import get_logger

logger = get_logger(__name__)

def convert_machine_to_extracted(raw_machine_dir: str, extracted_output_dir: str, year: str = "2024"):
    """
    Scans the deeply nested raw folders, dynamically standardizes the messy 
    CSV schemas, and converts them into pristine columnar Parquet files.
    """
    raw_path = Path(raw_machine_dir)
    extracted_path = Path(extracted_output_dir)
    extracted_path.mkdir(parents=True, exist_ok=True)

    for var_folder in raw_path.iterdir():
        if var_folder.is_dir():
            var_name = var_folder.name
            file_name = f"{year}_{var_name}.csv.xz"
            file_path = var_folder / file_name
            
            if file_path.exists():
                try:
                    with lzma.open(file_path, "rb") as f:
                        df = pl.read_csv(f.read())
                    
                    cols = df.columns
                    time_col = next((c for c in cols if c.lower() in ['wsdatetime', 'timestamp', 'time', 'date']), None)
                    
                    if not time_col:
                        logger.warning(f"Skipping {var_name}: No valid time column found in {cols}")
                        continue
                    
                    val_cols = [c for c in cols if c != time_col]
                    if val_cols:
                        val_col = val_cols[0]
                        df = df.rename({time_col: "WsDateTime", val_col: var_name})
                    
                    df = df.with_columns(
                        pl.col("WsDateTime").str.to_datetime(
                            format="%Y-%m-%d %H:%M:%S%.f",
                            time_unit="ms",
                            strict=True,
                        )
                    )
                    
                    parquet_out = extracted_path / f"{var_name}.parquet"
                    df.write_parquet(parquet_out)
                    
                except Exception as e:
                    logger.error(f"Failed processing {var_name}: {e}")

if __name__ == "__main__":
    logger.info("Starting robust extraction pipeline...")
    
    convert_machine_to_extracted(
        raw_machine_dir="data/raw/spark_raw/TEC_48S", 
        extracted_output_dir="data/raw/extracted/TEC_48S"
    )
    
    convert_machine_to_extracted(
        raw_machine_dir="data/raw/spark_raw/IPE_PV", 
        extracted_output_dir="data/raw/extracted/IPE_PV"
    )
    
    logger.info("Extraction layer complete with normalized schemas and ms-precision datetimes!")