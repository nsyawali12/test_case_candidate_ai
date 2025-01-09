def average_rating(books):
    # Ensure that there is at least one book to calculate the average
    if not books:
        return 0

    total_rating = 0
    book_count = 0

    # Summing up all ratings and counting books
    for book in books:
        total_rating += book.rating
        book_count += 1

    # Calculating the average rating
    if book_count == 0:
        return 0

    return total_rating / book_count
