def first_non_repeated_character(s):
    # Create a dictionary to store frequency count
    frequency = {}
    
    # Iterate through the string to count character frequencies
    for char in s:
        if char in frequency:
            frequency[char] += 1
        else:
            frequency[char] = 1
            
    # Iterate through the string again
    # Return the first character that has frequency count 1
    for char in s:
        if frequency[char] == 1:
            return char
    
    # If no such character is found, return None
    return None
