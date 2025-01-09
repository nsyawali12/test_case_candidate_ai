def calculate_avg_rating(books):
    total_ratings = 0
    total_books = 0

    for book in books:
        total_ratings += book.rating
        total_books += 1
    
    if total_books == 0:
        return 0
    
    avg_rating = total_ratings / total_books
    return avg_rating

# Example usage
books = Book.objects.all()
average_rating = calculate_avg_rating(books)
average_rating
