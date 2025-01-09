def calculate_average_rating(books):
    total_rating = 0
    num_books = 0

    for book in books:
        total_rating += book.rating
        num_books += 1

    if num_books == 0:
        return 0

    average_rating = total_rating / num_books
    return average_rating

# Example usage
books = Book.objects.all()
average_rating = calculate_average_rating(books)
average_rating
