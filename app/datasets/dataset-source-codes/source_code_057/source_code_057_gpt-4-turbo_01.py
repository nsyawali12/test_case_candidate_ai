import random

def roll_die():
    # Roll a six-sided die and return the result
    return random.randint(1, 6)

def roll_die_n_times(n):
    # Roll the die n times and count occurrences of each result
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for _ in range(n):
        result = roll_die()
        results[result] += 1
    return results
