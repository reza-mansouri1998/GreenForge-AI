import polars as pl
import pandera.polars as pa
from pathlib import Path
from greenforge_ai.utils.logger import get_logger

logger = get_logger(__name__)

# 1. Solar Photovoltaic Contract
PvTelemetrySchema = pa.DataFrameSchema(
    columns={
        "WsDateTime": pa.Column(pl.Datetime("ms"), unique=True),
        "AC_ActivePower": pa.Column(pl.Float64, checks=pa.Check.ge(0.0), nullable=False)
    },
    strict=False  
)

# 2. Heavy Manufacturing Contract (TEC_48S)
TecTelemetrySchema = pa.DataFrameSchema(
    columns={
        "WsDateTime": pa.Column(pl.Datetime("ms"), unique=True),
        # Ensure critical mechanical sensors are present and strictly numeric
        "Angle_U1": pa.Column(pl.Float64, nullable=False)
    },
    strict=False
)

def validate_gold_data(file_path: str, schema: pa.DataFrameSchema, system_name: str) -> None:
    """Loads the imputed Parquet file and runs the Pandera schema validation."""
    path = Path(file_path)
    if not path.exists():
        logger.error(f"Cannot validate missing file: {path}")
        return

    logger.info(f"Loading {system_name} for strict schema validation...")
    
    try:
        df = pl.read_parquet(path)
        schema.validate(df)
        logger.info(f"✅ {system_name} PASSED all physical and structural validations!")
        
    except pa.errors.SchemaError as e:
        logger.error(f"❌ {system_name} FAILED validation check!")
        logger.error(f"Failure details: {e}")
        raise

if __name__ == "__main__":
    logger.info("Starting Data Validation Pipeline...")
    
    validate_gold_data(
        file_path="data/processed/cleaned/IPE_PV_imputed.parquet",
        schema=PvTelemetrySchema,
        system_name="IPE_PV_Solar"
    )
    
    validate_gold_data(
        file_path="data/processed/cleaned/TEC_48S_imputed.parquet",
        schema=TecTelemetrySchema,
        system_name="TEC_48S_Manufacturing"
    )