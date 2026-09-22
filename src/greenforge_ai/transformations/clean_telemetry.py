import polars as pl
from pathlib import Path
from greenforge_ai.utils.logger import get_logger

logger = get_logger(__name__)

def clean_time_series(
    input_path: str, 
    output_path: str, 
    time_col: str = "WsDateTime", 
    interval: str = "5s"
) -> None:
    """
    Reads the consolidated wide table, enforces a strict chronological frequency, 
    and imputes missing sensor values using forward-fill.
    """
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    if not in_file.exists():
        logger.error(f"Input file not found: {in_file}")
        return

    logger.info(f"Loading {in_file.name} into memory...")
    
    try:
        # 1. Load and rigidly sort the wide Parquet file by time
        df = pl.read_parquet(in_file).sort(time_col)
        
        logger.info(f"Original shape: {df.shape}. Resampling to {interval} intervals...")
        
        # 2. Upsample to a continuous interval
        # This inserts empty rows for any time gaps where no sensor reported
        df_resampled = df.upsample(
            time_column=time_col, 
            every=interval,
            maintain_order=True
        )
        
        # 3. Impute the synthetic empty rows
        # Forward-fill assumes a sensor's state remains constant until a new reading arrives
        df_cleaned = df_resampled.with_columns(
            pl.all().exclude(time_col).fill_null(strategy="forward")
        )
        
        # 4. Save the finalized, gapless Gold-layer dataset
        df_cleaned.write_parquet(out_file)
        
        logger.info(f"Success! Cleaned shape: {df_cleaned.shape}. Saved to {out_file}")
        
    except Exception as e:
        logger.error(f"Failed to clean and resample data: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting Telemetry Cleaning & Resampling Pipeline...")
    
    clean_time_series(
        input_path="data/processed/cleaned/IPE_PV_wide.parquet",
        output_path="data/processed/cleaned/IPE_PV_imputed.parquet"
    )
    
    clean_time_series(
        input_path="data/processed/cleaned/TEC_48S_wide.parquet",
        output_path="data/processed/cleaned/TEC_48S_imputed.parquet"
    )