input_list = ["appel", "kroket", "kaassouflé", "schoen", "auto"]
def longest_word(input_list):
	longest_word = ""
	word_count = 0
	for word in input_list:
		if len(word) > word_count:
			longest_word = word
			word_count = len(word)
	return (longest_word, word_count)
longest_word(input_list)