def is_prime(n):
    # Corner cases
    if (n <= 1):
        return False
    if (n <= 3):
        return True

    # Check from 2 to n-1
    if(n % 2 == 0 or n % 3 == 0):
        return False

    i = 5
    while(i * i <= n):
        if (n % i == 0 or n % (i + 2) == 0):
            return False
        i += 6

    return True

# Test the function
print(is_prime(7))  # Should print: True
print(is_prime(10))  # Should print: False
