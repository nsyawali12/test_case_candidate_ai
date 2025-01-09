import datetime
from influxdb import InfluxDBClient

def compute_percentage_change(host='localhost', port=8086, database='', measurement='', field=''):

    try:
        client = InfluxDBClient(host, port)
        client.switch_database(database)
    except Exception as e:
        print(f"Failed to connect to the database: {e}")
        return None

    try:
        current_time = datetime.datetime.utcnow()

        one_hour_ago = current_time - datetime.timedelta(hours=1)
        two_hours_ago = current_time - datetime.timedelta(hours=2)

        query_latest = f"SELECT max({field}) FROM {measurement} WHERE time >= '{one_hour_ago}'"
        query_previous = f"SELECT max({field}) FROM {measurement} WHERE time >= '{two_hours_ago}' AND time < '{one_hour_ago}'"
    
        latest_result = client.query(query_latest).get_points().next()
        previous_result = client.query(query_previous).get_points().next()

        latest_value = latest_result['max']
        previous_value = previous_result['max']

        percentage_change = 100 * (latest_value - previous_value) / previous_value

        return f"The percentage change in {field} for the last hour is {percentage_change:.2f}%"

    except Exception as e:
        print(f"Failed to retrieve data: {e}")
        return None
