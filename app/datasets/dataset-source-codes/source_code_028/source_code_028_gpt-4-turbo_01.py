import influxdb
from datetime import datetime, timedelta

def get_percentage_change(database, measurement, field):
    """ Connects to InfluxDB, retrieves maximum value of a field from a measurement over the last hour,
        and calculates percentage change compared to the previous hour. """
    influx_client = None
    try:
        # Connect to InfluxDB
        influx_client = influxdb.InfluxDBClient('localhost', 8086, 'root', 'root', database)
        
        # Calculate time ranges for the current and previous hours
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=1)
        prev_hour_start_time = start_time - timedelta(hours=1)
        
        # Query to fetch the maximum value from the last hour
        query_current = f"SELECT max({field}) FROM {measurement} WHERE time >= '{start_time.isoformat()}Z' AND time < '{end_time.isoformat()}Z'"
        result_current = influx_client.query(query_current)
        max_current = list(result_current.get_points(measurement))[0].get('max')
        
        # Query to fetch the maximum value from the hour before the last hour
        query_previous = f"SELECT max({field}) FROM {measurement} WHERE time >= '{prev_hour_start_time.isoformat()}Z' AND time < '{start_time.isoformat()}Z'"
        result_previous = influx_client.query(query_previous)
        max_previous = list(result_previous.get_points(measurement))[0].get('max')
        
        # Calculate percentage change if both values are not None
        if max_current is not None and max_previous is not None and max_previous != 0:
            percentage_change = ((max_current - max_previous) / max_previous) * 100
        else:
            percentage_change = None
        
        return f"The percentage change in {field} for the last hour is {percentage_change}%"
    except Exception as e:
        return f"An error occurred: {str(e)}"
    finally:
        if influx_client is not None:
            influx_client.close()

# Example call
print(get_percentage_change('financial_data', 'stock_prices', 'closing_price'))
