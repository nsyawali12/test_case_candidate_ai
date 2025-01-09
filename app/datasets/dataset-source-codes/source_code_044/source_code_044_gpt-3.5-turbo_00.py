from myapp.models import Customer  # Import the Customer model from your Django app

def get_customers_with_zip_code_starting_with_123():
    customers = Customer.objects.filter(zip_code__startswith='123').values()  # Query to filter customers with zip_code starting with '123'
    return list(customers)  # Return a list of dictionaries for matching records
