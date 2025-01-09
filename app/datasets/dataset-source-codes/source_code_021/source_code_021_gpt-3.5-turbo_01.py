def first_non_repeated_char(input_str):
    char_count = {}
    
    for char in input_str:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    
    for char in input_str:
        if char_count[char] == 1:
            return char
    
    return None

# Example usage
print(first_non_repeated_char('abaccdeff'))  # Output: 'b'
print(first_non_repeated_char('abcdefggfedcba'))  # Output: None
