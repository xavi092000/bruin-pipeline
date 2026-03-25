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
    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    for col in float_cols:
        df[col] = pd.to_numeric(df[col], downcast="float")

    for col in int_cols:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    return df


def download_to_tempfile(session: requests.Session, url: str):
    for attempt in range(1, 4):
        try:
            response = session.get(url, stream=True, timeout=120)
            print(f"[ingestion.trips] GET {url} -> {response.status_code}")
        except Exception as exc:
            print(f"[ingestion.trips] request failed attempt {attempt}: {exc}")
            time.sleep(2)
            continue

        if response.status_code != 200:
            print(f"[ingestion.trips] skipping {url}, status={response.status_code}")
            return None

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")

        try:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if chunk:
                    tmp.write(chunk)
        finally:
            tmp.close()

        return tmp.name

    return None


def materialize():
    print("[ingestion.trips] materialize() start")

    start_date = os.environ["BRUIN_START_DATE"]
    end_date = os.environ["BRUIN_END_DATE"]

    print(f"[ingestion.trips] start={start_date}")
    print(f"[ingestion.trips] end={end_date}")

    bruin_vars = json.loads(os.environ.get("BRUIN_VARS", "{}"))
    taxi_types = bruin_vars.get("taxi_types", ["yellow"])

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        }
    )

    monthly_frames = []

    for taxi_type in taxi_types:
        for month_start in month_starts(start_date, end_date):
            year = month_start.year
            month = f"{month_start.month:02d}"

            url = (
                "https://d37ci6vzurychx.cloudfront.net/trip-data/"
                f"{taxi_type}_tripdata_{year}-{month}.parquet"
            )

            print(f"[ingestion.trips] processing {taxi_type} {year}-{month}")
            path = download_to_tempfile(session, url)

            if path is None:
                continue

            try:
                print(f"[ingestion.trips] reading parquet: {path}")
                df = pd.read_parquet(path, columns=DEFAULT_COLUMNS)
            finally:
                os.remove(path)

            print(f"[ingestion.trips] raw shape={df.shape}")

            df["taxi_type"] = taxi_type
            df = optimize_dataframe(df)

            print(f"[ingestion.trips] optimized shape={df.shape}")
            monthly_frames.append(df)

    if not monthly_frames:
        print("[ingestion.trips] no data loaded")
        return pd.DataFrame()

    print("[ingestion.trips] concatenating monthly frames")
    result_df = pd.concat(monthly_frames, ignore_index=True)

    print(f"[ingestion.trips] final shape={result_df.shape}")

    for col in result_df.select_dtypes(include=[np.number]).columns:
        result_df[col] = result_df[col].replace([np.inf, -np.inf], np.nan)

    return result_df