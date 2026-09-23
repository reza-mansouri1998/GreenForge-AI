import polars as pl
from pathlib import Path
import time
from greenforge_ai.utils.logger import get_logger

logger = get_logger(__name__)

def generate_interaction_features(
    input_path: str, 
    output_path: str, 
    smard_path: str = "data/raw/extracted/SMARD_DayAhead_Prices.parquet",
    dwd_path: str = "data/raw/extracted/DWD_Temperature.parquet"
) -> None:
    """
    Engineers cross-sensor interaction features and integrates 
    external energy prices (SMARD) and weather data (DWD) using an As-Of Join.
    """
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating interaction & external features for {in_file.name}...")
    start_time = time.time()
    
    try:
        df = pl.read_parquet(in_file).sort("WsDateTime")
        
        # 1. Integrate SMARD Day-Ahead Prices (Forcing exact 'ms' precision match)
        if Path(smard_path).exists():
            df_smard = pl.read_parquet(smard_path).with_columns(
                pl.col("WsDateTime").cast(pl.Datetime("ms"))
            ).sort("WsDateTime")
            df = df.join_asof(df_smard, on="WsDateTime", strategy="backward")
        else:
            logger.warning(f"SMARD data not found at {smard_path}")

        # 2. Integrate DWD Weather Data (Forcing exact 'ms' precision match)
        if Path(dwd_path).exists():
            df_dwd = pl.read_parquet(dwd_path).with_columns(
                pl.col("WsDateTime").cast(pl.Datetime("ms"))
            ).sort("WsDateTime")
            df = df.join_asof(df_dwd, on="WsDateTime", strategy="backward")
        else:
            logger.warning(f"DWD weather data not found at {dwd_path}")
            
        # 3. Domain-Specific Mathematical Interactions
        if "IPE_PV" in in_file.name:
            if "DC_Power" in df.columns:
                df = df.with_columns(
                    pl.when(pl.col("DC_Power") > 0)
                    .then(pl.col("AC_ActivePower") / pl.col("DC_Power"))
                    .otherwise(0.0)
                    .alias("inverter_efficiency")
                )
                
        elif "TEC_48S" in in_file.name:
            pass 
            
        df.write_parquet(out_file)
        elapsed = time.time() - start_time
        logger.info(f"✅ Interaction & external features saved to {out_file} in {elapsed:.2f}s.")
        
    except Exception as e:
        logger.error(f"Interaction feature generation failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting Final Feature Integration Pipeline...")
    generate_interaction_features(
        input_path="data/processed/features/IPE_PV_time_features.parquet", 
        output_path="data/processed/features/IPE_PV_final_features.parquet"
    )
    generate_interaction_features(
        input_path="data/processed/features/TEC_48S_time_features.parquet", 
        output_path="data/processed/features/TEC_48S_final_features.parquet"
    )