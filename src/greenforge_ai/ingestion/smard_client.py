# src/greenforge_ai/ingestion/smard_client.py
import requests
import polars as pl
from pathlib import Path
import time
from greenforge_ai.utils.logger import get_logger

logger = get_logger(__name__)

def fetch_smard_day_ahead_prices(output_path: str) -> None:
    """
    Fetches Day-Ahead Electricity Prices from the German SMARD API (Bundesnetzagentur).
    Module ID 4169: Day-Ahead Prices [€/MWh]
    Region: DE (Germany)
    Resolution: Hourly
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Connecting to SMARD API for Day-Ahead Prices (DE)...")
    start_time = time.time()
    
    # Corrected region code to "DE"
    index_url = "https://www.smard.de/app/chart_data/4169/DE/index_hour.json"
    
    try:
        response = requests.get(index_url, timeout=10)
        response.raise_for_status()
        timestamps = response.json()["timestamps"]
        
        if not timestamps:
            logger.error("No timestamp index returned from SMARD.")
            return
            
        latest_timestamp = timestamps[-1]
        
        # Corrected data URL structure: {filter}/{region}/{filterCopy}_{regionCopy}_{resolution}_{timestamp}.json
        data_url = f"https://www.smard.de/app/chart_data/4169/DE/4169_DE_hour_{latest_timestamp}.json"
        
        logger.info(f"Fetching price data for timestamp index: {latest_timestamp}...")
        data_response = requests.get(data_url, timeout=10)
        data_response.raise_for_status()
        
        series_data = data_response.json()["series"]
        
        df = pl.DataFrame(
        series_data, 
        schema=["unix_ms", "DayAhead_Price_EUR_MWh"], 
        orient="row"
        )
        
        df = df.with_columns(
            pl.from_epoch("unix_ms", time_unit="ms").alias("WsDateTime")
        ).drop("unix_ms")
        
        df = df.drop_nulls()
        
        df.write_parquet(out_file)
        elapsed = time.time() - start_time
        logger.info(f"✅ Downloaded {df.shape[0]} price records. Saved to {out_file} in {elapsed:.2f}s.")
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from SMARD API: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting SMARD Ingestion Pipeline...")
    fetch_smard_day_ahead_prices(
        output_path="data/raw/extracted/SMARD_DayAhead_Prices.parquet"
    )