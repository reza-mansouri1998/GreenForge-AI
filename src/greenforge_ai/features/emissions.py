"""Estimate grid-import CO₂ using published UBA annual factors."""

from __future__ import annotations

import logging
from typing import Mapping

import polars as pl

from greenforge_ai.utils.logger import get_logger

LOG = get_logger(__name__, channel="etl")

# Pinned publication: 2024 was estimated in the April 2025 release.
# Generation-related CO₂, not lifecycle CO₂-equivalent.
UBA_FACTORS_KG_PER_KWH = {2024: 0.363}

UBA_METADATA = {
    2024: {
        "publication_date": "2025-04-09",
        "source": (
            "https://www.umweltbundesamt.de/themen/"
            "co2-emissionen-pro-kilowattstunde-strom-2024"
        ),
        "status": "estimated",
        "method": "Annual generation-related CO2 factor",
    }
}


def add_emissions(
    frame: pl.DataFrame,
    *,
    energy_column: str = "import_kwh",
    date_column: str = "Date",
    output_column: str = "estimated_import_co2_kg",
    factors: Mapping[int, float] | None = None,
    timezone: str = "Europe/Berlin",
    logger: logging.Logger | None = None,
) -> pl.DataFrame:
    """Add estimated CO₂ in kg; leave the input frame unchanged.

    Energy must be non-negative grid-import energy in kWh.
    Dates use the local calendar year. Datetimes must be timezone-aware.
    Null energy stays null; missing dates and unsupported years raise errors.
    Custom factors must be verified annual factors in kg CO₂/kWh.
    """
    factors = dict(
        UBA_FACTORS_KG_PER_KWH if factors is None else factors
    )

    missing = {energy_column, date_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if output_column in frame.columns:
        raise ValueError(f"Output column already exists: {output_column}")
    if not frame.schema[energy_column].is_numeric():
        raise TypeError(f"{energy_column} must contain numeric kWh values.")

    date_type = frame.schema[date_column]
    date = pl.col(date_column)

    if date_type == pl.Date:
        year = date.dt.year()
    elif isinstance(date_type, pl.Datetime):
        if date_type.time_zone is None:
            raise ValueError("Datetime values must be timezone-aware.")
        year = date.dt.convert_time_zone(timezone).dt.year()
    else:
        raise TypeError(f"{date_column} must be Date or Datetime.")

    if frame[date_column].null_count():
        raise ValueError("Missing dates prevent year-matched calculations.")

    energy = pl.col(energy_column).cast(pl.Float64)
    invalid_count = frame.select(
        (
            energy.is_not_null()
            & (~energy.is_finite() | (energy < 0))
        ).sum()
    ).item()
    if invalid_count:
        raise ValueError(
            f"{energy_column}: {invalid_count:,} negative or non-finite values."
        )

    years = frame.select(year.alias("year"))["year"].unique().to_list()
    unsupported = sorted(set(years) - set(factors))
    if unsupported:
        raise ValueError(
            f"No verified emissions factor for years {unsupported}. "
            "Add sourced factors; do not reuse another year's value."
        )

    for data_year in years:
        value = float(factors[data_year])
        if not 0 <= value < float("inf"):
            raise ValueError(f"Invalid factor for {data_year}: {value}")
        factors[data_year] = value

    factor = year.replace_strict(
        factors, return_dtype=pl.Float64
    )
    result = frame.with_columns(
        (energy * factor).alias(output_column)
    )

    (logger if logger is not None else LOG).info(
        "CO2 estimate: rows=%d years=%s factors_kg_per_kwh=%s "
        "missing_energy=%d output=%s",
        frame.height,
        sorted(years),
        {year: factors[year] for year in sorted(years)},
        frame[energy_column].null_count(),
        output_column,
    )
    return result