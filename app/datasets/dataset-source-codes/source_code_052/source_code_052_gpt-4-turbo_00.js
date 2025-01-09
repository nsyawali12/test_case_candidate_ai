function buildSentence(words, maxChars) {
    // Initialize the result sentence with an empty string
    let sentence = "";
    // Initial used characters must account for the final full stop
    let usedChars = 1; // For the full stop at the end

    for (let word of words) {
        // Check if adding this word plus a space surpasses the limit
        if (usedChars + word.length + (sentence === "" ? 0 : 1) > maxChars) {
            break;
        }
        // Add the word to the sentence, with a space if it's not the first word
        sentence += (sentence === "" ? "" : " ") + word;
        // Update the count of used characters
        usedChars += word.length + (sentence === "" ? 0 : 1);
    }

    // Add a full stop only if there is at least one word
    return sentence.length > 0 ? sentence + '.' : '';
}
