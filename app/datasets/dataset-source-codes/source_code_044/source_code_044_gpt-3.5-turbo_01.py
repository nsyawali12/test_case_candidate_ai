from myapp.models import Customer

def get_customers_with_zip_code_starting_with_123():
    customers = Customer.objects.filter(zip_code__startswith='123').values()
    return list(customers)
