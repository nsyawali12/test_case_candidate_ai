def find_longest_word(word_list):
    # Initializing variables to track longest word and its length
    longest_word = ''
    longest_length = 0

    # Iterate over each word in the list
    for word in word_list:
        # If the length of current word is greater than longest_length, update longest_word and longest_length
        if len(word) > longest_length:
            longest_word = word
            longest_length = len(word)

    # Return the longest word and its length as a tuple
    return longest_word, longest_length
