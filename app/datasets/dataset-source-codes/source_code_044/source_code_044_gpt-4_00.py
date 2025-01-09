from django.core import serializers
from django.db.models import Q
from django.db.models import Value as V
from django.db.models.functions import Substr
from myapp.models import Customer

def get_customers_with_zip_code_starting_with_123():
    # Use Django ORM's filter method with the startswith field lookup to retrieve customers 
    # whose zip code starts with "123".
    customers = Customer.objects.filter(zip_code__startswith='123')

    # Convert the QuerySet to JSON
    json_customers = serializers.serialize('json', customers)

    # Deserialize the JSON to convert it into Python dictionaries
    customer_dicts = serializers.deserialize('json', json_customers)

    # Extract the object from each DeserializedObject and return the list
    return [customer_dict.object for customer_dict in customer_dicts]
