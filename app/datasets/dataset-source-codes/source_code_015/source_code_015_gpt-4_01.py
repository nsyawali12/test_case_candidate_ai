def sum_even_numbers(input_list):
    # If the list is empty, the sum is automatically 0
    if not input_list:
        return 0
    
    # Filter out the odd numbers from the list and sum the remaining even numbers
    return sum(num for num in input_list if num % 2 == 0)
