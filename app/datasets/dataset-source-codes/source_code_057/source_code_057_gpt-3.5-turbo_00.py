import random

def roll_die():
    return random.randint(1, 6)

def simulate_rolls(n):
    results = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    
    for _ in range(n):
        roll_result = roll_die()
        results[roll_result] += 1
        
    return results

n = 1000
simulation_result = simulate_rolls(n)
print(simulation_result)
