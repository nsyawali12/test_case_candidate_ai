from influxdb import InfluxDBClient
from datetime import datetime, timedelta

def get_percentage_change(measurement, field, database):
    try:
        client = InfluxDBClient(host='localhost', port=8086, database=database)
        client.switch_database(database)
        
        query = f"SELECT max({field}) FROM {measurement} WHERE time > now() - 2h"
        result = client.query(query)
        
        points = list(result.get_points())
        current_value = points[0]['max']
        previous_value = points[1]['max']
        
        percentage_change = ((current_value - previous_value) / previous_value) * 100
        
        return f"The percentage change in {field} for the last hour is {percentage_change:.2f}%"
        
    except Exception as e:
        return f"An error occurred: {e}"

# Example usage
measurement = 'stock_prices'
field = 'closing_price'
database = 'financial_data'

print(get_percentage_change(measurement, field, database))
