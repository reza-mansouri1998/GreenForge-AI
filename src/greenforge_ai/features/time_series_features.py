import polars as pl
from pathlib import Path
import time
from greenforge_ai.utils.logger import get_logger

logger = get_logger(__name__)

def generate_temporal_features(input_path: str, output_path: str, target_cols: list[str], time_col: str = "WsDateTime") -> None:
    """
    Engineers cyclical time features and rolling window metrics.
    """
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating time-series features for {in_file.name}...")
    start_time = time.time()
    
    try:
        lf = pl.scan_parquet(in_file)
        
        # 1. Cyclical Time Features
        lf = lf.with_columns([
            pl.col(time_col).dt.hour().alias("hour_of_day"),
            pl.col(time_col).dt.weekday().alias("day_of_week"),
            pl.col(time_col).dt.month().alias("month_of_year")
        ])
        
        # 2. Rolling Metrics & Lags (Based on strict 5s intervals = 180 rows per 15m)
        lf = lf.with_columns([
            pl.col(target_cols).rolling_mean(window_size=180).name.suffix("_roll_mean_15m"),
            pl.col(target_cols).rolling_std(window_size=180).name.suffix("_roll_std_15m"),
            pl.col(target_cols).shift(12).name.suffix("_lag_1m") # 1-minute lag
        ])
        
        # Drop the initial nulls created by the 180-row lookback
        lf = lf.drop_nulls()
        
        lf.sink_parquet(out_file)
        elapsed = time.time() - start_time
        logger.info(f"✅ Time-series features saved to {out_file} in {elapsed:.2f}s.")
        
    except Exception as e:
        logger.error(f"Time-series feature generation failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting Time-Series Feature Pipeline...")
    generate_temporal_features(
        input_path="data/processed/cleaned/IPE_PV_imputed.parquet", 
        output_path="data/processed/features/IPE_PV_time_features.parquet",
        target_cols=["AC_ActivePower"]
    )
    generate_temporal_features(
        input_path="data/processed/cleaned/TEC_48S_imputed.parquet", 
        output_path="data/processed/features/TEC_48S_time_features.parquet",
        target_cols=["Angle_U1"]
    )