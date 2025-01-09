def sum_of_evens(numbers):
    # Using list comprehension to filter even numbers and calculate their sum
    return sum([num for num in numbers if num % 2 == 0])
