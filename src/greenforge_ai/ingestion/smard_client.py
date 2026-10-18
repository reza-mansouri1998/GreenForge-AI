"""SMARD prices with bounded HTTP requests and retries."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from greenforge_ai.ingestion.dwd_client import HOUR, hourly_table
from greenforge_ai.utils.logger import LOG


def session():
    s = requests.Session()
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=4,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods={"GET"},
                respect_retry_after_header=False,
            )
        ),
    )
    s.headers["User-Agent"] = "GreenForge-ETL/2.0 (historical-energy-research)"
    return s


def get_json(s, url):
    with s.get(url, timeout=(10, 30), stream=True) as response:
        response.raise_for_status()
        chunks, size = [], 0
        for block in response.iter_content(65536):
            size += len(block)
            if size > 10 * 2**20:
                raise ValueError("External response exceeds 10 MiB")
            chunks.append(block)
    return json.loads(b"".join(chunks))


def fetch_smard(start, end, workers=2):
    base = "https://www.smard.de/app/chart_data/4169/DE"
    with session() as s:
        index = get_json(s, base + "/index_hour.json")
    targets = sorted({int(t) for t in index["timestamps"] if start - 14 * 24 * HOUR <= int(t) < end})
    if not targets or len(targets) > 400:
        raise ValueError("Unexpected SMARD index size")

    def chunk(t):
        with session() as s:
            rows = get_json(s, base + f"/4169_DE_hour_{t}.json")["series"]
        if len(rows) > 20000:
            raise ValueError("Unexpected SMARD chunk size")
        return rows

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, part in enumerate(pool.map(chunk, targets)):
            rows.extend(part)
            if len(rows) > 1_000_000:
                raise ValueError("Unexpected SMARD annual response size")
            LOG.info("SMARD chunk=%d/%d", i + 1, len(targets))
    return hourly_table(rows, start, end, "DayAhead_Price_EUR_MWh")
