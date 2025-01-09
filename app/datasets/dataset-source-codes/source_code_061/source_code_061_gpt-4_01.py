def find_longest_word(words_list):
    # Initialize the longest_word and its length to the first word in the list
    longest_word = words_list[0]
    max_length = len(longest_word)

    # Iterate over the rest of the words in the list
    for word in words_list[1:]:
        # If the current word's length is greater than max_length, update longest_word and max_length
        if len(word) > max_length:
            longest_word = word
            max_length = len(word) 

    # Return the longest_word and its length as a tuple
    return (longest_word, max_length)
