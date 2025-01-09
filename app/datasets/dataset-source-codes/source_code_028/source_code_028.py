from influxdb import InfluxDBCClient
from datetime import timedelta, datetime

def calculate_percentage_change(measurement, feild, database, host = 'localhost', port = 8086, username = None, password = None):
  try:
    # Connect to Influx DB.
    client = InfluxDBCClient(host, port, username, password, database)
    
    # Calculate the time range for the last two hours
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours = 2)

    # Build the Influx DB query
    query = f'SELECT MAX("{field}") FROM "{measurement}" WHERE time >= \'{start_time.strftime("%Y-%m-%dT%H:%M:%SZ")}\' AND time <= \'{end_time.strftime("%Y-%M-%dT%H:%M:%SZ")}\' GROUP BY time(1h)'

    # Query InfluxDB
    result = client.query(query)

    # Extract max value for the last two hours
    max_values = list(result.get_points())

    if len(max_values)>= 2:
      # Calculate percentage change
      current_hour_value = max_values[-1]['max']
      previous_hour_value = max_values[-2]['max']
      percentage_change = ((current_hour_value-previous_hour_value) / abs(previous_hour_value)) * 100
      return f'The percentage change in {feild} for the last hour is {percentage_change:.2f} %'
      
    else:
      return f'Insufficient data to calculate percentage change.'

  except Exception as e:
    return f'Error: {str(e)}'
    
  finally:
    if client:
      client.close()