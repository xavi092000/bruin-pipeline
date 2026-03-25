
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
    float_cols = df.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    int_cols = df.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

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
                return None
            time.sleep(attempt * 2)
            continue

        if response.status_code >= 500:
            print(
                f"[ingestion.trips] server error ({attempt}/3): "
                f"{response.status_code} for {url}"
            )
            if attempt == 3:
                return None
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
):
    parquet_file = pq.ParquetFile(parquet_path)
    schema_names = parquet_file.schema.names

    if selected_columns:
        columns = [c for c in selected_columns if c in schema_names]
    else:
        columns = schema_names

    print(f"[ingestion.trips] reading columns: {columns}")

    table = parquet_file.read(columns=columns)
    df = table.to_pandas()

    return df


def materialize():
    try:
        start_date = os.environ["BRUIN_START_DATE"]
        end_date = os.environ["BRUIN_END_DATE"]
    except KeyError as err:
        raise RuntimeError(
            "BRUIN_START_DATE and BRUIN_END_DATE must be set in the environment"
        ) from err

    print(f"[ingestion.trips] BRUIN_START_DATE={start_date}")
    print(f"[ingestion.trips] BRUIN_END_DATE={end_date}")

    bruin_vars = json.loads(os.environ.get("BRUIN_VARS", "{}"))
    taxi_types = bruin_vars.get("taxi_types", ["yellow"])
    selected_columns = bruin_vars.get("columns", DEFAULT_COLUMNS)

    # Guard against unavailable recent months (2 month lag)
    latest_available = (
        pd.Timestamp.today().replace(day=1) - relativedelta(months=2)
    )

    print(
        "[ingestion.trips] latest expected available month:",
        latest_available.strftime("%Y-%m"),
    )

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
                    f"[ingestion.trips] skipping unavailable month: "
                    f"{month_start.strftime('%Y-%m')}"
                )
                continue

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
        print("[ingestion.trips] no files loaded")
        return pd.DataFrame()

    print(f"[ingestion.trips] concatenating {len(dataframes)} dataframe(s)...")

    result_df = pd.concat(dataframes, ignore_index=True, copy=False)
    result_df = optimize_dataframe(result_df)

    print(f"[ingestion.trips] final dataframe shape: {result_df.shape}")
    print(f"[ingestion.trips] columns: {list(result_df.columns)}")

    nan_counts = result_df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"[ingestion.trips] NaN counts:\n{nan_counts[nan_counts > 0]}")

    inf_cols = []
    for col in result_df.select_dtypes(include=[np.number]).columns:
        if np.isinf(result_df[col]).any():
            inf_cols.append(col)

    if inf_cols:
        print(f"[ingestion.trips] infinity columns: {inf_cols}")

    return result_df