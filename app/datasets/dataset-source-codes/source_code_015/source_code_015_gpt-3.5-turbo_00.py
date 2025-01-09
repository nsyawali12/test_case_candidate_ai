def sum_of_even_numbers(lst):
    return sum(num for num in lst if num % 2 == 0)

# Example usage
input_list = [1, 2, 3, 4, 5]
print(sum_of_even_numbers(input_list))
