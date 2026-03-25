
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


# Garde seulement les colonnes utiles pour un projet taxi analytics
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
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)

    current = start
    while current <= end:
        yield current
        current = current + relativedelta(months=1)


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    int_like_cols = [
        "VendorID",
        "passenger_count",
        "RatecodeID",
        "PULocationID",
        "DOLocationID",
        "payment_type",
    ]
    float_like_cols = [
        "trip_distance",
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

    for col in int_like_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int32")

    for col in float_like_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce", downcast="float")

    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = df["store_and_fwd_flag"].astype("string")

    if "taxi_type" in df.columns:
        df["taxi_type"] = df["taxi_type"].astype("string")

    return df


def download_to_tempfile(session: requests.Session, url: str) -> str | None:
    response = None

    for attempt in range(1, 4):
        try:
            response = session.get(url, timeout=120, stream=True)
            print(f"[ingestion.trips] GET {url} -> {response.status_code}")
        except Exception as ex:
            print(f"[ingestion.trips] request error ({attempt}/3) for {url}: {ex}")
            if attempt == 3:
                raise
            time.sleep(attempt * 2)
            continue

        if response.status_code == 404:
            print(f"[ingestion.trips] not found: {url} (skipping)")
            return None

        if response.status_code == 403:
            print(f"[ingestion.trips] forbidden ({attempt}/3): {url}")
            if attempt == 3:
                raise RuntimeError(f"403 Forbidden for {url}")
            time.sleep(attempt * 2)
            continue

        if response.status_code >= 500:
            print(
                f"[ingestion.trips] server error ({attempt}/3): "
                f"{response.status_code} for {url}"
            )
            if attempt == 3:
                response.raise_for_status()
            time.sleep(attempt * 2)
            continue

        response.raise_for_status()
        break

    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp_path = tmp.name

    try:
        for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                tmp.write(chunk)
    finally:
        tmp.close()

    return tmp_path


def read_parquet_lightweight(
    parquet_path: str,
    selected_columns: list[str] | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = parquet_file.schema.names

    if selected_columns:
        columns = [c for c in selected_columns if c in schema_names]
    else:
        columns = schema_names

    print(f"[ingestion.trips] reading columns: {columns}")

    table = parquet_file.read(columns=columns)
    df = table.to_pandas()

    if max_rows is not None:
        df = df.head(max_rows).copy()
        print(f"[ingestion.trips] max_rows_per_file applied: {max_rows}")

    return df


def materialize():
    try:
        start_date = os.environ["BRUIN_START_DATE"]
        end_date = os.environ["BRUIN_END_DATE"]
    except KeyError as err:
        raise RuntimeError(
            "BRUIN_START_DATE and BRUIN_END_DATE must be set in the environment"
        ) from err

    bruin_vars = json.loads(os.environ.get("BRUIN_VARS", "{}"))
    taxi_types = bruin_vars.get("taxi_types", ["yellow"])
    selected_columns = bruin_vars.get("columns", DEFAULT_COLUMNS)
    max_rows_per_file = bruin_vars.get("max_rows_per_file")

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
            year = month_start.year
            month = f"{month_start.month:02d}"

            url = (
                f"https://d37ci6vzurychx.cloudfront.net/trip-data/"
                f"{taxi_type}_tripdata_{year}-{month}.parquet"
            )

            parquet_path = download_to_tempfile(session, url)
            if parquet_path is None:
                continue

            try:
                df = read_parquet_lightweight(
                    parquet_path=parquet_path,
                    selected_columns=selected_columns,
                    max_rows=max_rows_per_file,
                )
            finally:
                try:
                    os.remove(parquet_path)
                except OSError:
                    pass

            df["taxi_type"] = taxi_type
            df = optimize_dataframe(df)

            print(
                f"[ingestion.trips] loaded {taxi_type} {year}-{month} "
                f"shape={df.shape}"
            )

            dataframes.append(df)

    if not dataframes:
        print("[ingestion.trips] no files loaded for given date range/taxi types")
        return pd.DataFrame()

    print(f"[ingestion.trips] concatenating {len(dataframes)} dataframe(s)...")
    result_df = pd.concat(dataframes, ignore_index=True, copy=False)
    result_df = optimize_dataframe(result_df)

    print(f"[ingestion.trips] final dataframe shape: {result_df.shape}")
    print(f"[ingestion.trips] columns: {list(result_df.columns)}")
    print(f"[ingestion.trips] dtypes:\n{result_df.dtypes}")

    nan_counts = result_df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"[ingestion.trips] NaN counts per column:\n{nan_counts[nan_counts > 0]}")

    inf_cols = []
    for col in result_df.select_dtypes(include=[np.number]).columns:
        if np.isinf(result_df[col]).any():
            inf_cols.append(col)
    if inf_cols:
        print(f"[ingestion.trips] infinity values found in columns: {inf_cols}")

    return result_df