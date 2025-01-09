def is_prime(num):
    if num < 2:
        return False
    for i in range(2, int(num ** 0.5) + 1):
        if num % i == 0:
            return False
    return True

# Test the function with examples
print(is_prime(7))  # Output: True
print(is_prime(10))  # Output: False
