function makeSentence(words, maxChars) {
    // Start the sentence with an empty string
    let sentence = "";

    // Iterate over all words
    for(let i = 0; i < words.length; i++) {
        // Check if the word can fit in the sentence 
        // (considering the length of the word, space and the full stop)
        if(sentence.length + words[i].length + 1 <= maxChars) {
            // Add the word and a space to the sentence
            sentence += words[i] + " ";
        }
        else {
            // If the word doesn't fit, break the loop
            break;
        }
    }

    // If no words could fit in the sentence,
    // return an empty string with a full stop
    if(sentence == "") {
        return ".";
    }

    // Trim the last space and add a full stop to the sentence
    sentence = sentence.trim() + ".";

    // Return the prepared sentence
    return sentence;
}
