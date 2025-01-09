import random

def roll_die():
    return random.randint(1, 6)

def simulate_rolls(n):
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    for _ in range(n):
        result = roll_die()
        results[result] += 1
    
    return results

# Example usage
n = 1000
output = simulate_rolls(n)
print(output)
