def sum_even_numbers(numbers):
    return sum(num for num in numbers if num % 2 == 0)

# Example usage
input_list = [1, 2, 3, 4, 5]
result = sum_even_numbers(input_list)
print(result)
