def first_non_repeated_char(s):
    # Dictionary to store count of each character
    char_count = {}
    
    # Count each character in the string
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1

    # Find the first character with a count of 1
    for char in s:
        if char_count[char] == 1:
            return char

    # Return None if no non-repeated character is found
    return None
