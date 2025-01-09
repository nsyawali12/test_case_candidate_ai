def find_longest_word(words):
    max_length = 0
    longest_word = ''

    for word in words:
        if len(word) > max_length:
            max_length = len(word)
            longest_word = word

    return longest_word, max_length

# Test cases
print(find_longest_word(['hello', 'world', 'python', 'programming']))  # Output: ('programming', 11)
print(find_longest_word(['short', 'longer', 'longest', 'size']))  # Output: ('longest', 7)
