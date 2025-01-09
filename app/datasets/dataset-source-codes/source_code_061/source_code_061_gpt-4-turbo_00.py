def find_longest_word(word_list):
    # Initialize the default variable for the longest word and its length
    longest_word = ''
    max_length = 0
    
    # Loop through each word in the list
    for word in word_list:
        # Check if the current word's length is greater than the max length found so far
        if len(word) > max_length:
            longest_word = word
            max_length = len(word)
    
    # Return the longest word and its length as a tuple
    return (longest_word, max_length)
