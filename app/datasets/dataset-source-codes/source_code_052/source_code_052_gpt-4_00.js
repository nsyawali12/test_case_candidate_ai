function constructSentence(words, maxChars) {
    // start with an empty sentence
    let sentence = '';

    // go through each word
    for (let word of words) {
        // try to fit the word
        if ((sentence + word).length + 1 <= maxChars) {
            sentence += word + ' ';
        } else {
            // if we cannot fit any more words, break loop
            break;
        }
    }

    // remove last space and add a full stop
    if (sentence) {
        sentence = sentence.trim() + '.';
    }

    return sentence;
}

let words = ['hello', 'world', 'this', 'is', 'a', 'test'];
let maxChars = 20;
console.log(constructSentence(words, maxChars));  // Returns: 'hello world this.'
