"""@bruin
name: ingestion.trips

type: python

image: python:3.11

connection: duckdb-default

materialization:
  type: table
  strategy: create+replace
@bruin"""

import json
import os
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import requests
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
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)

    current = start
    while current <= end:
        yield current
        current = current + relativedelta(months=1)


def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    print("[ingestion.trips] optimize_dataframe() start")
    print(f"[ingestion.trips] shape before optimize: {df.shape}")

    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    print(f"[ingestion.trips] float64 cols: {list(float_cols)}")
    print(f"[ingestion.trips] int64 cols: {list(int_cols)}")

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    print("[ingestion.trips] optimize_dataframe() done")
    return df


def download_to_tempfile(session: requests.Session, url: str):
    print(f"[ingestion.trips] download_to_tempfile() start for {url}")

    for attempt in range(1, 4):
        print(f"[ingestion.trips] attempt {attempt}/3 for {url}")

        try:
            response = session.get(url, stream=True, timeout=120)
            print(f"[ingestion.trips] GET {url} -> {response.status_code}")
        except Exception as exc:
            print(f"[ingestion.trips] request failed on attempt {attempt}: {exc}")
            time.sleep(2)
            continue

        if response.status_code != 200:
            print(f"[ingestion.trips] skipping {url}, status={response.status_code}")
            return None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        print(f"[ingestion.trips] writing response to temp file: {tmp.name}")

        total_bytes = 0
        chunk_count = 0

        try:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    chunk_count += 1
                    total_bytes += len(chunk)
                    tmp.write(chunk)
                    print(
                        f"[ingestion.trips] wrote chunk {chunk_count}, "
                        f"size={len(chunk)} bytes, total={total_bytes} bytes"
                    )
        finally:
            tmp.close()

        print(
            f"[ingestion.trips] download complete for {url}, "
            f"tempfile={tmp.name}, total_bytes={total_bytes}"
        )
        return tmp.name

    print(f"[ingestion.trips] all attempts failed for {url}")
    return None


def read_parquet_chunked(path: str, columns):
    print(f"[ingestion.trips] read_parquet_chunked() start for {path}")
    print(f"[ingestion.trips] requested columns: {columns}")

    parquet = pq.ParquetFile(path)
    print(f"[ingestion.trips] num_row_groups={parquet.num_row_groups}")

    dfs = []

    for i in range(parquet.num_row_groups):
        print(f"[ingestion.trips] reading row group {i + 1}/{parquet.num_row_groups}")
        table = parquet.read_row_group(i, columns=columns)
        print(
            f"[ingestion.trips] row group {i + 1} table rows={table.num_rows}, "
            f"cols={table.num_columns}"
        )
        df = table.to_pandas()
        print(f"[ingestion.trips] row group {i + 1} df shape={df.shape}")
        dfs.append(df)

    result = pd.concat(dfs, ignore_index=True)
    print(f"[ingestion.trips] read_parquet_chunked() done, final shape={result.shape}")
    return result


def materialize():
    print("[ingestion.trips] materialize() start")

    start_date = os.environ["BRUIN_START_DATE"]
    end_date = os.environ["BRUIN_END_DATE"]

    print(f"[ingestion.trips] start={start_date}")
    print(f"[ingestion.trips] end={end_date}")

    bruin_vars = json.loads(os.environ.get("BRUIN_VARS", "{}"))
    taxi_types = bruin_vars.get("taxi_types", ["yellow"])

    print(f"[ingestion.trips] BRUIN_VARS={bruin_vars}")
    print(f"[ingestion.trips] taxi_types={taxi_types}")

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        }
    )
    print("[ingestion.trips] requests session initialized")

    dataframes = []

    for taxi_type in taxi_types:
        print(f"[ingestion.trips] processing taxi_type={taxi_type}")

        for month_start in month_starts(start_date, end_date):
            year = month_start.year
            month = f"{month_start.month:02d}"

            url = (
                "https://d37ci6vzurychx.cloudfront.net/trip-data/"
                f"{taxi_type}_tripdata_{year}-{month}.parquet"
            )

            print(f"[ingestion.trips] processing month={year}-{month}")
            print(f"[ingestion.trips] url={url}")

            path = download_to_tempfile(session, url)
            if path is None:
                print(f"[ingestion.trips] no file downloaded for {year}-{month}, skipping")
                continue

            try:
                print(f"[ingestion.trips] starting parquet read for {path}")
                df = read_parquet_chunked(path, DEFAULT_COLUMNS)
                print(f"[ingestion.trips] parquet read complete for {year}-{month}")
            finally:
                print(f"[ingestion.trips] removing temp file {path}")
                os.remove(path)

            print(f"[ingestion.trips] adding taxi_type column={taxi_type}")
            df["taxi_type"] = taxi_type

            df = optimize_dataframe(df)

            print(
                f"[ingestion.trips] loaded {taxi_type} {year}-{month} "
                f"shape={df.shape}"
            )

            dataframes.append(df)
            print(f"[ingestion.trips] dataframes collected so far: {len(dataframes)}")

    if not dataframes:
        print("[ingestion.trips] no data loaded")
        return pd.DataFrame()

    print("[ingestion.trips] concatenating all monthly dataframes")
    result_df = pd.concat(dataframes, ignore_index=True)

    print(f"[ingestion.trips] final dataframe shape: {result_df.shape}")
    print(f"[ingestion.trips] columns: {list(result_df.columns)}")
    print(f"[ingestion.trips] dtypes:\n{result_df.dtypes}")
    print("[ingestion.trips] checking for NaN/inf values...")

    nan_counts = result_df.isna().sum()
    if nan_counts.sum() > 0:
        print(f"[ingestion.trips] NaN counts per column:\n{nan_counts[nan_counts > 0]}")
    else:
        print("[ingestion.trips] no NaN values found")

    inf_cols = []
    for col in result_df.select_dtypes(include=[np.number]).columns:
        if np.isinf(result_df[col]).any():
            inf_cols.append(col)

    if inf_cols:
        print(f"[ingestion.trips] infinity values found in columns: {inf_cols}")
    else:
        print("[ingestion.trips] no infinity values found")

    print("[ingestion.trips] materialize() done")
    return result_df