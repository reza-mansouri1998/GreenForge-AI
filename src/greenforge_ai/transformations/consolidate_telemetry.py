import polars as pl
from pathlib import Path
import time
from greenforge_ai.utils.logger import get_logger
from greenforge_ai.utils.exceptions import MissingDataError

logger = get_logger(__name__)

def build_wide_telemetry_table(
    extracted_dir: str, 
    output_path: str, 
    base_table_name: str, 
    time_col: str = "WsDateTime"
) -> None:
    """
    Executes an iterative, memory-bound Polars merge to consolidate single-sensor 
    Parquet files into a wide analytical table.
    """
    path = Path(extracted_dir)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    all_files = list(path.glob("*.parquet"))
    join_files = [f for f in all_files if f.stem != base_table_name]
    
    if not join_files:
        logger.warning(f"No additional files found in {extracted_dir} to join.")
        return

    base_table_path = path / f"{base_table_name}.parquet"
    if not base_table_path.exists():
        raise MissingDataError(f"Base table {base_table_name}.parquet is missing!")

    logger.info(f"Starting iterative Polars merge for {len(join_files) + 1} files in {path.name}...")
    start_time = time.time()
    
    try:
        main_df = pl.read_parquet(base_table_path).unique(subset=[time_col], keep="first")
        
        for i, file in enumerate(join_files, 1):
            df_join = pl.read_parquet(file).unique(subset=[time_col], keep="first")
            main_df = main_df.join(df_join, on=time_col, how="left", coalesce=True)
            
            if i % 30 == 0 or i == len(join_files):
                logger.info(f"Merged {i}/{len(join_files)} files...")
                
        main_df.write_parquet(out_file)
        
        elapsed = time.time() - start_time
        logger.info(f"Success! Final shape {main_df.shape}. Saved to {out_file} in {elapsed:.2f}s.")
        
    except Exception as e:
        logger.error(f"Iterative merge failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting Data Consolidation Pipeline...")
    
    build_wide_telemetry_table(
        extracted_dir="data/raw/extracted/IPE_PV",
        output_path="data/processed/cleaned/IPE_PV_wide.parquet",
        base_table_name="AC_ActivePower"
    )
    
    build_wide_telemetry_table(
        extracted_dir="data/raw/extracted/TEC_48S",
        output_path="data/processed/cleaned/TEC_48S_wide.parquet",
        base_table_name="Angle_U1"
    )