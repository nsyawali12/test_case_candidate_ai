def sum_even_numbers(input_list):
    # Initialize sum
    sum = 0
    # Loop through each number in the list
    for number in input_list:
        # Check if the number is even
        if number % 2 == 0:
            # If it is, add it to the sum
            sum += number
    # Return the sum
    return sum
