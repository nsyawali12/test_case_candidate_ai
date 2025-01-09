from django.views.generic import ListView

class PersonListView(ListView):
    model = Person
    template_name = 'person_list.html'

    def get_queryset(self):
        return Person.objects.filter(birthdate__year__lte=2005)