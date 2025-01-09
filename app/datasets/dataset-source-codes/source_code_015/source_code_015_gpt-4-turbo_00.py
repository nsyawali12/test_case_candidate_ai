def sum_of_evens(numbers):
    # Sum all even numbers in the input list
    return sum(num for num in numbers if num % 2 == 0)
