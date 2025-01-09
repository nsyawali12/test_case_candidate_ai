import random

# Function to simulate rolling a six-sided die
def roll_die():
    return random.randint(1, 6)

# Function to simulate rolling the die n times
def roll_n_times(n):
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for _ in range(n):
        roll_result = roll_die()
        results[roll_result] += 1
    return results
