def calculate_average_rating(books):
    # Handle the case where the list of books is empty
    if not books:
        return 0

    total_rating = 0
    book_count = 0

    # Loop through each book object to sum up the ratings
    for book in books:
        total_rating += book.rating
        book_count += 1
    
    # Calculate and return the average rating
    if book_count == 0:
        return 0
    else:
        return total_rating / book_count
