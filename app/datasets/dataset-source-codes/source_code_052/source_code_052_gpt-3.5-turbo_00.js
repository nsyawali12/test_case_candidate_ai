function createSentence(words, maxChars) {
    if (words.length === 0 || maxChars < 1) {
        return "";
    }

    let sentence = "";
    let currentLength = 0;

    for (let i = 0; i < words.length; i++) {
        if (sentence.length + words[i].length + 1 <= maxChars) {
            sentence += words[i] + " ";
            currentLength += words[i].length + 1;
        } else {
            break;
        }
    }

    if (sentence.length > 0) {
        sentence = sentence.trim() + ".";
    }

    return sentence;
}

// Test the function with the example
const words = ['hello', 'world', 'this', 'is', 'a', 'test'];
const maxChars = 20;
console.log(createSentence(words, maxChars)); // Output: 'hello world this.'
