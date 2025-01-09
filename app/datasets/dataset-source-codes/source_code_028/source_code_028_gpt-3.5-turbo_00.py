from influxdb import InfluxDBClient
from datetime import datetime, timedelta

def calculate_percentage_change(measurement, field, database):
    try:
        # Connect to InfluxDB database
        client = InfluxDBClient(host='localhost', port=8086, database=database)
        client.switch_database(database)

        # Calculate time range for last hour
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)

        # Query InfluxDB for maximum value for the given field in the last hour
        result = client.query("SELECT MAX({}) FROM {} WHERE time >= '{}' AND time <= '{}'".format(field, measurement, start_time.strftime('%Y-%m-%dT%H:%M:%SZ'), end_time.strftime('%Y-%m-%dT%H:%M:%SZ')))
        max_value_last_hour = list(result.get_points())[0]['max']

        # Query InfluxDB for maximum value for the given field in the hour before
        prev_end_time = start_time
        prev_start_time = prev_end_time - timedelta(hours=1)
        prev_result = client.query("SELECT MAX({}) FROM {} WHERE time >= '{}' AND time <= '{}'".format(field, measurement, prev_start_time.strftime('%Y-%m-%dT%H:%M:%SZ'), prev_end_time.strftime('%Y-%m-%dT%H:%M:%SZ')))
        max_value_prev_hour = list(prev_result.get_points())[0]['max']

        # Calculate percentage change
        if max_value_prev_hour != 0:
            percentage_change = ((max_value_last_hour - max_value_prev_hour) / max_value_prev_hour) * 100
        else:
            percentage_change = 0

        return "The percentage change in {} for the last hour is {:.1f}%".format(field, percentage_change)

    except Exception as e:
        return "Error: {}".format(e)

# Example usage
measurement = 'stock_prices'
field = 'closing_price'
database = 'financial_data'
print(calculate_percentage_change(measurement, field, database))
