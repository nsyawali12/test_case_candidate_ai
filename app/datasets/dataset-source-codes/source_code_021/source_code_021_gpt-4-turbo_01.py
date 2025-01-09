def first_non_repeated_char(s):
    # Dictionary to count occurrences of each character
    char_count = {}
    
    # Populate the dictionary with character counts
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
            
    # Find the first character with a count of 1
    for char in s:
        if char_count[char] == 1:
            return char
            
    return None
