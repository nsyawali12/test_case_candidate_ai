def move_zeros_to_end(lst):
    zero_count = 0
    for i in range(len(lst)):
        if lst[i] != 0:
            lst[i], lst[zero_count] = lst[zero_count], lst[i]
            zero_count += 1

# Test cases
lst1 = [0, 1, 0, 3, 12]
move_zeros_to_end(lst1)
print(lst1)  # Output: [1, 3, 12, 0, 0]

lst2 = [0, 0, 1]
move_zeros_to_end(lst2)
print(lst2)  # Output: [1, 0, 0]
