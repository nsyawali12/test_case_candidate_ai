def first_non_repeated_char(string):
    char_count = {}
    non_repeated_chars = set()

    for char in string:
        if char in char_count:
            non_repeated_chars.discard(char)
        else:
            char_count[char] = 1
            non_repeated_chars.add(char)

    for char in string:
        if char in non_repeated_chars:
            return char

    return None

# Example usage
result = first_non_repeated_char('abaccdeff')
print(result)  # Output: 'b'

result = first_non_repeated_char('abcdefggfedcba')
print(result)  # Output: None