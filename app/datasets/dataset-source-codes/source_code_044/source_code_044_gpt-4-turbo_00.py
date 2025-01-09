from django.db.models import Q

def get_customers_by_zip_prefix():
    # Query Customer models where the zip_code starts with '123'
    customers = Customer.objects.filter(zip_code__startswith='123')
    
    # Convert the QuerySet to a list of dictionaries
    customer_list = list(customers.values())
    
    return customer_list
