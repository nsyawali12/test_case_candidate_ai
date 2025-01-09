def avg_rating_of_books():
  books = Books.objects.all()
  total_rating = 0
  for book in books:
    total_rating += book.rating

  total_books = books.count()
  average_rating_of_books = total_rating/total_books
  return average_rating_of_books