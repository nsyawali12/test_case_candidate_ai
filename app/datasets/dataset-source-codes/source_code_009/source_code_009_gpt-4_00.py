def calculate_average_rating():
    # Get all the book objects
    all_books = Book.objects.all()

    # If no books return 0
    if not all_books:
        return 0

    # Initialize sum as 0
    total_rating = 0

    # Iterate over all_books to sum all the ratings
    for book in all_books:
        total_rating += book.rating

    # Calculate and return average rating
    return total_rating / len(all_books)
