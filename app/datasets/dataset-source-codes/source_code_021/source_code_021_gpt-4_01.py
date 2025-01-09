def firstNonRepeatingChar(string):
    # Initializing a dictionary to keep track of character counts
    count_dict = {}

    # Iterating through the string and counting characters
    for char in string:
        if char in count_dict:
            count_dict[char] += 1
        else:
            count_dict[char] = 1

    # Iterating through the string again and returning the first character with a count of 1
    for char in string:
        if count_dict[char] == 1:
            return char

    # If all characters are repeated, return None
    return None
