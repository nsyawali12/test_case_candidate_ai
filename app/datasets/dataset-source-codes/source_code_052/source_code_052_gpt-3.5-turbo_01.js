function generateSentence(words, maxChars) {
    if (words.length === 0 || maxChars < 1) {
        return '';
    }

    let sentence = '';
    let i = 0;

    while (i < words.length && sentence.length + words[i].length + 1 <= maxChars) {
        sentence += words[i] + ' ';
        i++;
    }

    // Trim the extra space at the end and add a full stop
    sentence = sentence.trim() + '.';
    
    return sentence;
}

// Example usage
const words = ['hello', 'world', 'this', 'is', 'a', 'test'];
const maxChars = 20;
console.log(generateSentence(words, maxChars)); // Output: 'hello world this.'
