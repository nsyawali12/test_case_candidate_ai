import random

# Function to simulate rolling a six-sided die
def roll_die():
    return random.randint(1, 6)

# Function that takes a number n as input and simulates rolling the die n times
def simulate_rolls(n):
    # Initializing a dictionary to hold counts
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    for _ in range(n):
        # Incrementing the count of the rolled number
        counts[roll_die()] += 1
    return counts
