
"""Trip ingestion helpers for the taxi pipeline."""

import json
import os
import time
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def month_starts(start_date: str, end_date: str):
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)

    current = start
    while current <= end:
        yield current
        current = current + relativedelta(months=1)


def build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=2,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive",
        }
    )

    return session


def fetch_parquet(session: requests.Session, url: str) -> pd.DataFrame | None:
    for attempt in range(1, 4):
        try:
            response = session.get(url, timeout=120)
        except requests.RequestException as ex:
            print(f"[ingestion.trips] request error ({attempt}/3) for {url}: {ex}")
            if attempt == 3:
                raise
            time.sleep(attempt * 2)
            continue

        status = response.status_code
        print(f"[ingestion.trips] GET {url} -> {status}")

        if status == 404:
            print(f"[ingestion.trips] not found: {url} (skipping)")
            return None

        if status == 403:
            print(f"[ingestion.trips] forbidden ({attempt}/3) for {url}")
            if attempt == 3:
                response.raise_for_status()
            time.sleep(attempt * 2)
            continue

        if status >= 500:
            print(f"[ingestion.trips] server error ({attempt}/3) {status} for {url}")
            if attempt == 3:
                response.raise_for_status()
            time.sleep(attempt * 2)
            continue

        response.raise_for_status()

        try:
            return pd.read_parquet(BytesIO(response.content))
        except Exception as ex:
            print(f"[ingestion.trips] parquet read error for {url}: {ex}")
            raise

    return None


def materialize() -> pd.DataFrame:
    try:
        start_date = os.environ["BRUIN_START_DATE"]
        end_date = os.environ["BRUIN_END_DATE"]
    except KeyError as err:
        raise RuntimeError(
            "BRUIN_START_DATE and BRUIN_END_DATE must be set in the environment"
        ) from err

    bruin_vars = json.loads(os.environ.get("BRUIN_VARS", "{}"))
    taxi_types = bruin_vars.get("taxi_types", ["yellow"])

    dataframes = []
    session = build_session()

    for taxi_type in taxi_types:
        for month_start in month_starts(start_date, end_date):
            year = month_start.year
            month = f"{month_start.month:02d}"

            url = (
                "https://d37ci6vzurychx.cloudfront.net/trip-data/"
                f"{taxi_type}_tripdata_{year}-{month}.parquet"
            )

            df = fetch_parquet(session, url)
            if df is None:
                continue

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
    if nan_counts.any():
        print(f"[ingestion.trips] NaN counts per column:\n{nan_counts[nan_counts > 0]}")

    inf_cols = [
        col
        for col in result_df.select_dtypes(include=[np.number]).columns
        if np.isinf(result_df[col]).any()
    ]
    if inf_cols:
        print(f"[ingestion.trips] infinity values found in columns: {inf_cols}")

    datetime_cols = result_df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
    if len(datetime_cols) > 0:
        print(f"[ingestion.trips] datetime columns detected: {list(datetime_cols)}")

    return result_df
