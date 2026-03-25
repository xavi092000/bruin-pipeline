
name: ingestion.trips

type: python

image: python:3.11

connection: duckdb-default

materialization:
  type: table
  strategy: create+replace

# Python entrypoint for taxi ingestion
import json
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta


def month_starts(start_date: str, end_date: str):
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)

    current = start
    while current <= end:
        yield current
        current = current + relativedelta(months=1)


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

            response = None

            for attempt in range(1, 4):
                try:
                    response = session.get(url, timeout=120)
                    print(f"[ingestion.trips] GET {url} -> {response.status_code}")
                except Exception as ex:
                    print(f"[ingestion.trips] request error ({attempt}/3) for {url}: {ex}")
                    if attempt == 3:
                        raise
                    time.sleep(attempt * 2)
                    continue

                if response.status_code == 404:
                    print(f"[ingestion.trips] not found: {url} (skipping)")
                    response = None
                    break

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

            if response is None:
                continue

            df = pd.read_parquet(BytesIO(response.content))
            df["taxi_type"] = taxi_type
            dataframes.append(df)

    if not dataframes:
        print("[ingestion.trips] no files loaded for given date range/taxi types")
        return pd.DataFrame()

    result_df = pd.concat(dataframes, ignore_index=True)

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