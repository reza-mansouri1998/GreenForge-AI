import polars as pl
from pathlib import Path
import time
from greenforge_ai.utils.logger import get_logger

logger = get_logger(__name__)

def generate_interaction_features(input_path: str, output_path: str) -> None:
    """
    Engineers cross-sensor interaction features (e.g., efficiency ratios, deltas).
    Reads the output of the time-series script to layer features sequentially.
    """
    in_file = Path(input_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating interaction features for {in_file.name}...")
    start_time = time.time()
    
    try:
        lf = pl.scan_parquet(in_file)
        
        # Apply domain-specific mathematical interactions
        if "IPE_PV" in in_file.name:
            # Example: Inverter Efficiency (AC Power / DC Power)
            # Using pl.when().then() to avoid division by zero
            if "DC_Power" in lf.collect_schema().names():
                lf = lf.with_columns(
                    pl.when(pl.col("DC_Power") > 0)
                    .then(pl.col("AC_ActivePower") / pl.col("DC_Power"))
                    .otherwise(0.0)
                    .alias("inverter_efficiency")
                )
                
        elif "TEC_48S" in in_file.name:
            # Placeholder for manufacturing differentials (e.g., Target vs Actual)
            # lf = lf.with_columns((pl.col("Angle_U1_Target") - pl.col("Angle_U1")).alias("angle_u1_error"))
            pass
            
        lf.sink_parquet(out_file)
        elapsed = time.time() - start_time
        logger.info(f"✅ Interaction features saved to {out_file} in {elapsed:.2f}s.")
        
    except Exception as e:
        logger.error(f"Interaction feature generation failed: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting Interaction Feature Pipeline...")
    generate_interaction_features(
        input_path="data/processed/features/IPE_PV_time_features.parquet", 
        output_path="data/processed/features/IPE_PV_final_features.parquet"
    )
    generate_interaction_features(
        input_path="data/processed/features/TEC_48S_time_features.parquet", 
        output_path="data/processed/features/TEC_48S_final_features.parquet"
    )