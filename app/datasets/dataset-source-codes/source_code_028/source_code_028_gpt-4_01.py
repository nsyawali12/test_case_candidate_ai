import influxdb
from datetime import datetime, timedelta

def calculate_percentage_change(db_name, measurement, field):
    client = None
    try:
        client = influxdb.InfluxDBClient(database=db_name)
    except Exception as e:
        print(f"Error while connecting to the database: {e}")
        return

    now = datetime.utcnow()
    previous_hour = now - timedelta(hours=1)
    two_hours_ago = now - timedelta(hours=2)

    try:
        # Query for the max value over the last hour
        result = client.query(f"SELECT max({field}) FROM {measurement} WHERE time >= '{previous_hour}' AND time < '{now}'")
        max_value_last_hour = list(result.get_points(measurement=measurement))[0]['max']

        # Query for the max value during the previous hour
        result = client.query(f"SELECT max({field}) FROM {measurement} WHERE time >= '{two_hours_ago}' AND time < '{previous_hour}'")
        max_value_previous_hour = list(result.get_points(measurement=measurement))[0]['max']
    except Exception as e:
        print(f"Error while retrieving data from the database: {e}")
        return

    # Calculate percentage change
    percentage_change = ((max_value_last_hour - max_value_previous_hour) / max_value_previous_hour) * 100

    return percentage_change
