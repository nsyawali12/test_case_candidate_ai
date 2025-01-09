def is_prime(num):
    # Handle special cases
    if num <= 1:
        return False
    if num <= 3:
        return True
    
    # Check for even numbers and multiples of 3
    if num % 2 == 0 or num % 3 == 0:
        return False
    
    # Use the 6k +/- 1 rule to check for factors up to the square root of num
    i = 5
    while i * i <= num:
        if num % i == 0 or num % (i + 2) == 0:
            return False
        i += 6
        
    return True

# Example usage
print(is_prime(7))  # Output: True
print(is_prime(10)) # Output: False
