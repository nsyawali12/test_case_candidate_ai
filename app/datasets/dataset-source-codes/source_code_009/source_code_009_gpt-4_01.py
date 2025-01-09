def average_rating():
    books = Book.objects.all()
    
    total_rating = 0
    total_books = len(books)
    
    for book in books:
        total_rating += book.rating
    
    # We avoid division by zero in case there are no books
    if total_books > 0:
        return total_rating / total_books
    else:
        return 0
