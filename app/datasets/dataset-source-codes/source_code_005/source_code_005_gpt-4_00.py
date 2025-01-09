def is_prime(n):
    # Check if the number is less than 2 (not prime) or if it's not a whole number
    if n < 2 or not isinstance(n, int):
        return False
    # Check if the number is 2 (the first prime number)
    elif n == 2:
        return True
    # Check if the number is divisible by any other number
    else:
        for i in range(2, n):
            if n % i == 0:
                return False
        return True

# Example usage:
print(is_prime(7))  # True
print(is_prime(10)) # False
