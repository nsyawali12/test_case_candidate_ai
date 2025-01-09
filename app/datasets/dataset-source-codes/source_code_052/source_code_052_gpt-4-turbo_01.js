function createSentence(words, maxChars) {
    // Handling edge case when the input array is empty or no word can fit due to maxChars being too low
    if (words.length === 0 || maxChars <= 0) {
        return ".";
    }

    let sentence = "";
    let currentLength = 0;

    for (let word of words) {
        // Check if the current word can fit with an extra space (if it's not the first word) and a full stop
        if (currentLength + word.length + (currentLength > 0 ? 1 : 0) > maxChars - 1) {
            break;
        }

        // Add space before the word if it's not the first word
        if (currentLength > 0) {
            sentence += " ";
            currentLength += 1;  // account for the space
        }

        // Add the word to the sentence
        sentence += word;
        currentLength += word.length;
    }

    // Append a full stop at the end
    sentence += ".";

    return sentence;
}
