def is_prime(num):
    # Check if the number is less than 2
    if num < 2:
        return False
    # Check for factors from 2 to the square root of num
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            return False
    return True

# Example usage
print(is_prime(7))  # Output: True
print(is_prime(10)) # Output: False
