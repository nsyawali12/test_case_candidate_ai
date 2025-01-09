import random

def roll_die():
    """Simulate rolling a six-sided die using random.randint."""
    return random.randint(1, 6)

def roll_die_n_times(n):
    """Simulate rolling a six-sided die n times and return a count of each result."""
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for _ in range(n):
        result = roll_die()
        results[result] += 1
    return results
