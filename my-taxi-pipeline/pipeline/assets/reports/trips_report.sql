/* @bruin
name: reports.trips_report
type: duckdb.sql

depends:
  - staging.trips

materialization:
  type: table
@bruin */

SELECT
    CAST(tpep_pickup_datetime AS DATE) AS trip_date,
    taxi_type,
    payment_type,
    COUNT(*) AS trip_count,
    SUM(fare_amount) AS total_fare,
    AVG(fare_amount) AS avg_fare
FROM staging.trips
GROUP BY 1, 2, 3