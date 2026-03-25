"""@bruin

name: test.hello

type: python

image: python:3.11

connection: duckdb-default

materialization:
  type: table
  strategy: create+replace
@bruin"""

import pandas as pd


def materialize():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "message": ["hello", "bruin", "test"],
        }
    )