from django.db.models import Q

def get_customers_by_zip_prefix():
    # Filtering customers whose zip_code starts with '123'
    customers = Customer.objects.filter(zip_code__startswith='123')
    
    # Creating list of dictionaries with all fields from each customer record
    customer_data_list = list(customers.values())
    
    return customer_data_list
