from django.core.serializers import serialize
import json
from .models import Customer

def get_customers():
    # Use Django Filter with istartswith to get all Customers starts with '123'
    customers = Customer.objects.filter(zip_code__istartswith='123')

    # Serialize the queryset to JSON
    serialized_customers = serialize('json', customers)

    # Convert the serialized data to list of dictionaries
    customer_list = json.loads(serialized_customers)
    
    # Loop through the list to replace the model name 'pk' with 'id' for each dictionary
    for customer in customer_list:
        customer['id'] = customer.pop('pk')
        
    return customer_list
