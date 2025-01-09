import influxdb
import datetime

def get_percentage_change(measurement, field, database_url='http://localhost:8086', database_name='example_db'):
    # Create a client and connect to the InfluxDB instance
    client = influxdb.InfluxDBClient(host=database_url, port=8086)
    
    try:
        # Connect to the specified database
        client.switch_database(database_name)
        
        # Calculate the time range for the last two hours
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(hours=2)
        
        # Query for maximum value of the given field from the specified measurement over the last two one-hour periods
        query = f"""
        SELECT MAX("{field}") FROM "{measurement}"
        WHERE time >= '{start_time.isoformat()}Z' AND time < '{end_time.isoformat()}Z'
        GROUP BY time(1h) fill(none)
        """
        
        # Execute the query
        results = client.query(query)
        
        # Extract the maximum values from the query results
        max_values = list(results.get_points())

        if len(max_values) != 2:
            return "Insufficient data for calculation"
        
        # Obtain the maximum values for the last two hours
        max_prev_hour = max_values[0]['max']
        max_last_hour = max_values[1]['max']
        
        # Calculate the percentage change between the two hours
        if max_prev_hour is None or max_last_hour is None:
            return "Insufficient data for calculation"
        
        percentage_change = ((max_last_hour - max_prev_hour) / max_prev_hour) * 100 if max_prev_hour != 0 else float('inf')
        
        return f"The percentage change in {field} for the last hour is {percentage_change:.2f}%"
    
    except influxdb.exceptions.InfluxDBClientError as e:
        return f"Database error: {str(e)}"
    
    except Exception as e:
        return f"An error occurred: {str(e)}"
    
    finally:
        client.close()

# Example usage (uncomment to test in a real environment):
# print(get_percentage_change('stock_prices', 'closing_price', 'http://localhost:8086', 'financial_data'))
