import os, json
from io import BytesIO
import pandas as pd, requests
from dateutil.relativedelta import relativedelta


def month_starts(start_date, end_date):
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)
    current = start
    while current < end:
        yield current
        current += relativedelta(months=1)


def main():
    os.environ['BRUIN_START_DATE'] = '2019-01-01'
    os.environ['BRUIN_END_DATE'] = '2019-03-01'
    os.environ['BRUIN_VARS'] = json.dumps({'taxi_types': ['yellow', 'green', 'fhv']})

    start_date = os.environ['BRUIN_START_DATE']
    end_date = os.environ['BRUIN_END_DATE']
    bruin_vars = json.loads(os.environ.get('BRUIN_VARS', '{}'))
    taxi_types = bruin_vars.get('taxi_types', ['yellow'])

    print('settings', start_date, end_date, taxi_types)

    dataframes = []
    for taxi_type in taxi_types:
        for month_start in month_starts(start_date, end_date):
            year = month_start.year
            month = f"{month_start.month:02d}"
            url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year}-{month}.parquet'
            print('GET', url)
            r = requests.get(url, timeout=60)
            print('status', r.status_code)
            r.raise_for_status()
            df = pd.read_parquet(BytesIO(r.content))
            print('df', taxi_type, year, month, len(df))
            dataframes.append(df)

    print('frames', len(dataframes))


if __name__ == '__main__':
    main()