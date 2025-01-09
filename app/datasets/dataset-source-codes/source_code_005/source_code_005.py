def is_prime(number):
    if number <= 1:
        return False  # 1 and all negative numbers are not prime

    if number <= 3:
        return True  # 2 and 3 are prime

    if number % 2 == 0:
        return False  # Even numbers greater than 2 are not prime

    # Check for divisibility by odd numbers from 3 to the square root of the number
    for i in range(3, int(number**0.5) + 1, 2):
        if number % i == 0:
            return False  # The number is divisible by 'i' and hence not prime

    return True  # If no divisors were found, the number is prime

# Test cases
input1 = 7
output1 = is_prime(input1)
print(output1)  # Output: True

input2 = 10
output2 = is_prime(input2)
print(output2)  # Output: False