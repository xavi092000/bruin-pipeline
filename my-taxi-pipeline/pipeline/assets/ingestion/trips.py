name: ingestion.trips

type: python

image: python:3.11

connection: duckdb-default

materialization:
  type: table
  strategy: create+replace

import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import requests
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta


DEFAULT_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "RatecodeID",
    "store_and_fwd_flag",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "Airport_fee",
]


def month_starts(start_date: str, end_date: str):
    """Yield the first day of each month between start_date and end_date."""
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)

    current = start
    while current <= end:
        yield current
        current = current + relativedelta(months=1)


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize dataframe dtypes to reduce memory usage."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def download_to_tempfile(session: requests.Session, url: str) -> str | None:
    """Download parquet file from URL to temporary file."""
    for attempt in range(3):
        try:
            response = session.get(url, stream=True, timeout=120)
            print(f"[ingestion.trips] GET {url} -> {response.status_code}")

        except Exception as e:
            print(f"[ingestion.trips] retry download {e}")
            time.sleep(2)
            continue

        if response.status_code != 200:
            print(f"[ingestion.trips] skipping {url}")
            return None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")

        for chunk in response.iter_content(8 * 1024 * 1024):
            tmp.write(chunk)

        tmp.close()

        return tmp.name

    return None


def read_parquet_chunked(path: str, columns: list[str]) -> pd.DataFrame:
    """Read parquet file by row groups for memory efficiency."""
    parquet = pq.ParquetFile(path)

    dfs = []

    for i in range(parquet.num_row_groups):
        print(
            f"[ingestion.trips] reading row group {i + 1}/{parquet.num_row_groups}"
        )

        table = parquet.read_row_group(i, columns=columns)
        df = table.to_pandas()
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def materialize() -> pd.DataFrame:
    """Fetch and combine taxi trip data from multiple parquet files."""
    try:
        start_date = os.environ["BRUIN_START_DATE"]
        end_date = os.environ["BRUIN_END_DATE"]
    except KeyError as err:
        raise RuntimeError(
            "BRUIN_START_DATE and BRUIN_END_DATE must be set in the environment"
        ) from err

    print(f"[ingestion.trips] start={start_date}")
    print(f"[ingestion.trips] end={end_date}")

    bruin_vars = json.loads(os.environ.get("BRUIN_VARS", "{}"))
    taxi_types = bruin_vars.get("taxi_types", ["yellow"])

    latest_available = (
        pd.Timestamp.today().replace(day=1) - relativedelta(months=2)
    )
    print(f"[ingestion.trips] latest available: {latest_available.strftime('%Y-%m')}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        }
    )

    dataframes = []

    for taxi_type in taxi_types:
        for month_start in month_starts(start_date, end_date):
            if month_start > latest_available:
                print(
                    f"[ingestion.trips] skip recent: {month_start.strftime('%Y-%m')}"
                )
                continue

            year = month_start.year
            month = f"{month_start.month:02d}"

            url = (
                "https://d37ci6vzurychx.cloudfront.net/trip-data/"
                f"{taxi_type}_tripdata_{year}-{month}.parquet"
            )

            path = download_to_tempfile(session, url)
            if path is None:
                continue

            try:
                df = read_parquet_chunked(path, DEFAULT_COLUMNS)
            finally:
                os.remove(path)

            df["taxi_type"] = taxi_type
            df = optimize_dataframe(df)

            print(
                f"[ingestion.trips] loaded {taxi_type} {year}-{month} shape={df.shape}"
            )
            dataframes.append(df)

    if not dataframes:
        print("[ingestion.trips] no data loaded")
        return pd.DataFrame()

    result = pd.concat(dataframes, ignore_index=True)
    print(f"[ingestion.trips] final shape: {result.shape}")

    return result