import polars as pl
from pathlib import Path
import time
from greenforge_ai.utils.logger import get_logger
from wetterdienst.provider.dwd.observation import DwdObservationRequest

logger = get_logger(__name__)

def fetch_dwd_weather_data(output_path: str, station_id: str = "01420") -> None:
    """
    Fetches historical hourly weather data (Temperature) from DWD (Deutscher Wetterdienst).
    Default station: 01420 (Frankfurt/Main - central industrial hub).
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Connecting to DWD API for Station {station_id}...")
    start_time = time.time()
    
    try:
        request = DwdObservationRequest(
            parameters=[("hourly", "temperature_air")],
            periods=["recent"]
        ).filter_by_station_id(station_id=[station_id])
        
        values_df = request.values.all().df.to_pandas()
        df = pl.from_pandas(values_df)
        
        # FIX: Cast the Categorical column to String before applying .str.contains()
        df = df.filter(pl.col("parameter").cast(pl.String).str.contains("temperature_air_mean"))
        
        df = df.with_columns(
            pl.col("date").dt.cast_time_unit("ms").alias("WsDateTime"),
            pl.col("value").alias("Air_Temperature_C")
        ).select(["WsDateTime", "Air_Temperature_C"])
        
        df = df.drop_nulls()
        
        df.write_parquet(out_file)
        elapsed = time.time() - start_time
        logger.info(f"✅ Downloaded {df.shape[0]} DWD weather records. Saved to {out_file} in {elapsed:.2f}s.")
        
    except Exception as e:
        logger.error(f"Failed to fetch data from DWD API: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting DWD Weather Ingestion Pipeline...")
    fetch_dwd_weather_data(
        output_path="data/raw/extracted/DWD_Temperature.parquet"
    )