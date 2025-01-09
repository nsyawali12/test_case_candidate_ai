def find_longest_word(words):
    # Initialize the longest word and its length
    longest_word = ""
    max_length = 0
    
    # Iterate over the list of words
    for word in words:
        # Check if the current word's length is greater than the max_length found so far
        if len(word) > max_length:
            longest_word = word
            max_length = len(word)
    
    # Return a tuple containing the longest word and its length
    return (longest_word, max_length)
